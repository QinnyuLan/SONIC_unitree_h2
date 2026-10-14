"""Repo-native LoRA adapters for SONIC H2LoRA adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from loguru import logger
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer plus trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        input_slice: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.input_slice = input_slice
        adapter_in = base.in_features if input_slice is None else input_slice[1] - input_slice[0]
        if adapter_in <= 0:
            raise ValueError(f"Invalid LoRA input slice {input_slice} for {base.in_features=}")
        self.lora_a = nn.Parameter(torch.empty(rank, adapter_in))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_x = x if self.input_slice is None else x[..., self.input_slice[0] : self.input_slice[1]]
        adapter_x = self.dropout(adapter_x)
        lora_out = F.linear(F.linear(adapter_x, self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out


@dataclass
class LoRAStats:
    trainable_params: int
    total_params: int
    trainable_ratio: float
    wrapped_modules: list[str]

    def as_dict(self) -> dict[str, int | float | list[str]]:
        return {
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_ratio": self.trainable_ratio,
            "wrapped_modules": self.wrapped_modules,
        }


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _iter_linear_children(module: nn.Module) -> list[tuple[str, nn.Module, str, nn.Linear]]:
    linears: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for parent_name, parent in module.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                linears.append((full_name, parent, child_name, child))
    return linears


def _wrap_linear(
    parent: nn.Module,
    child_name: str,
    child: nn.Linear,
    rank: int,
    alpha: float,
    dropout: float,
    input_slice: tuple[int, int] | None = None,
) -> None:
    _replace_child(parent, child_name, LoRALinear(child, rank, alpha, dropout, input_slice))


def _token_flattened_dim(actor: nn.Module) -> int:
    actor_module = actor.actor_module
    return int(actor_module.token_dim * actor_module.max_num_tokens)


def apply_sonic_h2lora(
    policy: nn.Module,
    value_model: nn.Module | None,
    lora_cfg,
) -> LoRAStats:
    """Freeze SONIC and add LoRA modules following the Any2Any S7 split.

    For the current SONIC MLP layout this means:
    - actor ``g1_dyn`` first linear: LoRA only on the proprioception input
      slice, keeping reference/token input frozen;
    - actor ``g1_dyn`` hidden and output linear layers: LoRA;
    - encoders, FSQ, reference/kinematic decoder and base weights: frozen;
    - critic hidden backbone linears: LoRA, input and value projections frozen.
    """
    if not bool(lora_cfg.get("enabled", True)):
        return LoRAStats(0, sum(p.numel() for p in policy.parameters()), 0.0, [])

    rank = int(lora_cfg.get("rank", 32))
    alpha = float(lora_cfg.get("alpha", rank))
    dropout = float(lora_cfg.get("dropout", 0.0))
    max_trainable_ratio = float(lora_cfg.get("max_trainable_ratio", 0.08))

    if value_model is not None:
        _freeze_module(value_model)
    _freeze_module(policy)

    wrapped: list[str] = []

    actor_module = policy.actor_module
    if "g1_dyn" not in actor_module.decoders:
        raise KeyError("Expected actor_module.decoders['g1_dyn'] for H2LoRA adaptation")
    dyn_decoder = actor_module.decoders["g1_dyn"]
    if not hasattr(dyn_decoder, "module"):
        raise TypeError("g1_dyn decoder must expose a BaseModule.module MLP")

    dyn_linears = _iter_linear_children(dyn_decoder.module)
    if len(dyn_linears) < 2:
        raise ValueError(f"Expected at least 2 Linear layers in g1_dyn, got {len(dyn_linears)}")

    token_dim = _token_flattened_dim(policy)
    first_name, first_parent, first_child_name, first_child = dyn_linears[0]
    if first_child.in_features <= token_dim:
        raise ValueError(
            "g1_dyn first linear input is smaller than token_flattened dim: "
            f"{first_child.in_features=} {token_dim=}"
        )
    _wrap_linear(
        first_parent,
        first_child_name,
        first_child,
        rank,
        alpha,
        dropout,
        input_slice=(token_dim, first_child.in_features),
    )
    wrapped.append(f"actor.g1_dyn.{first_name}:proprioception_slice")

    for name, parent, child_name, child in dyn_linears[1:]:
        _wrap_linear(parent, child_name, child, rank, alpha, dropout)
        wrapped.append(f"actor.g1_dyn.{name}")

    if value_model is not None:
        critic_backbone = value_model.critic_module
        if not hasattr(critic_backbone, "module"):
            raise TypeError("critic backbone must expose BaseModule.module MLP")
        critic_linears = _iter_linear_children(critic_backbone.module)
        if len(critic_linears) < 3:
            raise ValueError(f"Expected critic input/hidden/output Linear layers, got {len(critic_linears)}")
        for name, parent, child_name, child in critic_linears[1:-1]:
            _wrap_linear(parent, child_name, child, rank, alpha, dropout)
            wrapped.append(f"critic.backbone.{name}")

    total = sum(p.numel() for module in (policy, value_model) if module is not None for p in module.parameters())
    trainable = sum(
        p.numel()
        for module in (policy, value_model)
        if module is not None
        for p in module.parameters()
        if p.requires_grad
    )
    ratio = trainable / max(total, 1)
    if ratio > max_trainable_ratio:
        raise RuntimeError(
            f"H2LoRA trainable ratio {ratio:.4%} exceeds configured limit {max_trainable_ratio:.4%}"
        )
    logger.info(
        f"Applied H2LoRA: rank={rank}, alpha={alpha}, dropout={dropout}, "
        f"trainable={trainable}/{total} ({ratio:.3%}), modules={len(wrapped)}"
    )
    return LoRAStats(trainable, total, ratio, wrapped)

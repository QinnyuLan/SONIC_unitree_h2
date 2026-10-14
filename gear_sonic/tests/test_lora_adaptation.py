import torch
from torch import nn

from gear_sonic.trl.utils.lora import LoRALinear, apply_sonic_h2lora


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(10, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 29),
        )


class DummyActorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_dim = 2
        self.max_num_tokens = 2
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": DummyDecoder(),
                "g1_kin": DummyDecoder(),
            }
        )


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_module = DummyActorModule()
        self.std = nn.Parameter(torch.ones(29))


class DummyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic_module = DummyDecoder()


def test_lora_linear_preserves_shape_and_freezes_base():
    base = nn.Linear(5, 7)
    lora = LoRALinear(base, rank=2, alpha=2, input_slice=(1, 4))
    out = lora(torch.randn(3, 5))
    assert out.shape == (3, 7)
    assert not base.weight.requires_grad
    assert lora.lora_a.requires_grad
    assert lora.lora_b.requires_grad


def test_apply_sonic_h2lora_freezes_non_lora_params():
    policy = DummyPolicy()
    value = DummyValue()
    stats = apply_sonic_h2lora(
        policy,
        value,
        {"enabled": True, "rank": 2, "alpha": 2, "dropout": 0.0, "max_trainable_ratio": 1.0},
    )
    trainable_names = [
        name
        for module_name, module in (("policy", policy), ("value", value))
        for name, param in module.named_parameters()
        if param.requires_grad
        for name in [f"{module_name}.{name}"]
    ]
    assert trainable_names
    assert all(("lora_a" in name or "lora_b" in name) for name in trainable_names)
    assert policy.std.requires_grad is False
    assert stats.trainable_params > 0
    assert any("proprioception_slice" in name for name in stats.wrapped_modules)

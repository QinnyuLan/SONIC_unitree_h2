# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).
# Adapted for SONIC_unitree_h2: audio+text inference only; see licenses/echo_g/.
"""The released ECHO-G network and eight-step Euler inference, without training code."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import validate_conditions


def modulate(values, shift, scale):
    return values * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def masked_softmax(logits, padding):
    mask = padding[:, None, None, :]
    safe = torch.where((~mask).any(dim=-1, keepdim=True), logits.masked_fill(mask, -torch.inf), 0.)
    return safe.softmax(dim=-1).masked_fill(mask, 0.)


class WordTimeBlock(nn.Module):
    def __init__(self, hidden, heads, feedforward):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.self_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.cross_output = nn.Linear(hidden, hidden)
        self.mlp_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden, feedforward), nn.GELU(), nn.Linear(feedforward, hidden))
        self.adaptive_norm = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))
        # All parameters, including the bounded attention parameters, load from EMA.
        self.qk_temperature_raw = nn.Parameter(torch.zeros(heads))
        self.word_sigma_raw = nn.Parameter(torch.zeros(heads))
        self.word_lag_raw = nn.Parameter(torch.zeros(heads))
        self.word_mix_raw = nn.Parameter(torch.zeros(heads))

    def cross_attention(self, motion, text, padding, distance):
        batch, frames, hidden = motion.shape
        tokens = text.shape[1]
        query = self.query(motion).view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(text).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(text).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        q = F.normalize(query.float(), dim=-1, eps=1e-6)
        k = F.normalize(key.float(), dim=-1, eps=1e-6)
        temperature = 1. + 15. * self.qk_temperature_raw.float().sigmoid()
        sigma = .25 + 1.75 * self.word_sigma_raw.float().sigmoid()
        lag = .5 * self.word_lag_raw.float().tanh()
        mix = .5 * self.word_mix_raw.float().sigmoid()
        logits = (q @ k.transpose(-1, -2)) * temperature[None, :, None, None]
        global_weights = masked_softmax(logits, padding)
        standardized = (distance[:, None].float() - lag[None, :, None, None]) / sigma[None, :, None, None]
        prior = -.5 * standardized.square()
        local_weights = masked_softmax(logits + prior.clamp_min(-12.), padding)
        support = prior.exp().masked_fill(padding[:, None, None, :], 0.).amax(dim=-1, keepdim=True)
        effective_mix = mix[None, :, None, None] * support
        weights = (1 - effective_mix) * global_weights + effective_mix * local_weights
        attended = (weights @ value.float()).to(motion.dtype)
        return self.cross_output(attended.transpose(1, 2).reshape(batch, frames, hidden))

    def forward(self, values, condition, text, motion_padding, text_padding, distance):
        parameters = self.adaptive_norm(condition).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = parameters[:3]
        cross_shift, cross_scale, cross_gate = parameters[3:6]
        mlp_shift, mlp_scale, mlp_gate = parameters[6:]
        hidden = modulate(self.self_norm(values), self_shift, self_scale)
        attended, _ = self.self_attention(hidden, hidden, hidden,
                                           key_padding_mask=motion_padding, need_weights=False)
        values = values + self_gate.unsqueeze(1) * attended
        hidden = modulate(self.cross_norm(values), cross_shift, cross_scale)
        values = values + cross_gate.unsqueeze(1) * self.cross_attention(hidden, text, text_padding, distance)
        hidden = modulate(self.mlp_norm(values), mlp_shift, mlp_scale)
        return values + mlp_gate.unsqueeze(1) * self.mlp(hidden)


class SpeechGroundedDiT(nn.Module):
    """Architecture of the pinned audio+text checkpoint; parameter names match EMA."""

    def __init__(self, config):
        super().__init__()
        expected = dict(motion_dim=39, max_t=608, pos_enc='learned', hidden=768,
                        layers=12, heads=8, ff_dim=2048, dropout=0.)
        if config != expected:
            raise ValueError('Expected the released ECHO-G audio+text model configuration')
        hidden = config['hidden']
        self.hidden = hidden
        self.motion_projection = nn.Linear(39, hidden)
        self.audio_norm = nn.LayerNorm(1024)
        self.audio_projection = nn.Linear(1024, hidden)
        self.text_norm = nn.LayerNorm(2560)
        self.text_projection = nn.Linear(2560, hidden)
        self.timestep_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.zeros(1, config['max_t'], hidden))
        self.blocks = nn.ModuleList([WordTimeBlock(hidden, config['heads'], config['ff_dim'])
                                     for _ in range(config['layers'])])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_adaptive_norm = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.output = nn.Linear(hidden, 39)

    def forward(self, motion, flow_time, audio, text, motion_mask, text_padding, distance):
        frames = motion.shape[1]
        if not 2 <= frames <= 600 or not 1 <= text.shape[1] <= 256:
            raise ValueError('ECHO-G supports 2–600 frames and 1–256 text tokens')
        half = self.hidden // 2
        frequencies = torch.exp(-math.log(10_000.) * torch.arange(
            half, device=flow_time.device, dtype=torch.float32) / half)
        angles = flow_time.float()[:, None] * frequencies[None] * 1_000.
        condition = self.timestep_mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        hidden = (self.motion_projection(motion) + self.audio_projection(self.audio_norm(audio))
                  + self.position[:, :frames].to(dtype=motion.dtype))
        text = self.text_projection(self.text_norm(text))
        for block in self.blocks:
            hidden = block(hidden, condition, text, ~motion_mask, text_padding, distance)
        shift, scale = self.final_adaptive_norm(condition).chunk(2, dim=-1)
        return self.output(modulate(self.final_norm(hidden), shift, scale))


@torch.no_grad()
def predict(checkpoint_path, condition, *, seed=0, device='cpu'):
    """Single utterance, EMA weights, eight Euler steps and guidance scale 1."""
    validate_conditions(condition)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    experiment = checkpoint['experiment']
    if (experiment.get('target'), experiment.get('conditioning'), experiment.get('fps')) != ('robot', 'audio-text', 30):
        raise ValueError('Expected a 30 Hz robot audio+text checkpoint')
    model = SpeechGroundedDiT(checkpoint['model_config'])
    model.load_state_dict(checkpoint['ema'], strict=True)
    model.to(device).eval()
    mean = torch.as_tensor(checkpoint['motion_mean']).float().reshape(-1).to(device)
    std = torch.as_tensor(checkpoint['motion_std']).float().reshape(-1).to(device)
    if (mean.shape != (39,) or std.shape != (39,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid checkpoint normalization statistics')
    del checkpoint
    audio = condition['audio_features'].float().unsqueeze(0).to(device)
    text = condition['text_tokens'].float().unsqueeze(0).to(device)
    frames = audio.shape[1]
    # Compute frame/token distances on CPU before transfer, matching upstream collation.
    centers = condition['token_times'].float().mean(dim=-1)
    distance = (torch.arange(frames, dtype=torch.float32)[:, None] / 30 - centers[None, :])[None].to(device)
    motion_mask = torch.ones((1, frames), dtype=torch.bool, device=device)
    text_padding = torch.zeros((1, text.shape[1]), dtype=torch.bool, device=device)
    digest = hashlib.sha256(f"{condition['stem']}:{seed}".encode()).digest()
    generator = torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], 'little') % (2**31 - 1))
    sample = torch.randn((frames, 39), generator=generator, device=device).unsqueeze(0)
    for step in range(8):
        flow_time = torch.full((1,), step / 8, device=device, dtype=torch.float32)
        sample = sample + .125 * model(sample, flow_time, audio, text, motion_mask, text_padding, distance)
    physical = (sample[0].float() * std + mean).cpu().contiguous()
    if not torch.isfinite(physical).all():
        raise ValueError('Generated motion contains non-finite values')
    return physical

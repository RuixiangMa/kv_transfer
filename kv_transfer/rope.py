"""RoPE strip/re-apply utilities for cross-model KV cache transfer.

Qwen3 uses GPT-NeoX style RoPE (half rotation). The forward rotation is:
    k_rot = k * cos + rotate_half(k) * sin
The inverse (stripping) is:
    k = k_rot * cos - rotate_half(k_rot) * sin
R_Theta is orthogonal, so inversion is exact at negligible cost.
"""
import torch


def rotate_half(x):
    """Rotates half the hidden dims: [-x2, x1] (GPT-NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_cos_sin(head_dim, rope_theta, seq_len, device, dtype=torch.float32):
    """Compute cos/sin tables for RoPE.

    Matches transformers Qwen3RotaryEmbedding.forward:
        inv_freq = 1 / (theta ^ (arange(0, dim, 2) / dim))
        freqs = outer(positions, inv_freq)  # [seq_len, dim//2]
        emb = cat(freqs, freqs)             # [seq_len, dim]
        cos, sin = emb.cos(), emb.sin()

    Returns: (cos, sin) each of shape [seq_len, head_dim]
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)    # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Apply forward RoPE to keys.

    Args:
        x: [batch, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
    Returns: same shape as x
    """
    cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def strip_rope(x, cos, sin):
    """Strip (invert) RoPE from keys.

    Args:
        x: [batch, num_heads, seq_len, head_dim] (post-RoPE keys)
        cos, sin: [seq_len, head_dim]
    Returns: same shape as x (pre-RoPE keys)
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos - rotate_half(x) * sin

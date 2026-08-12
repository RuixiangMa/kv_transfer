"""Unit test: RoPE strip + re-apply roundtrip."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from kv_transfer.rope import compute_rope_cos_sin, apply_rope, strip_rope, rotate_half


def test_rope_roundtrip():
    """strip_rope(apply_rope(x)) should recover x within tolerance."""
    head_dim = 128
    rope_theta = 1000000
    seq_len = 1024
    batch = 2
    num_heads = 8
    device = "cuda"

    x = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    cos, sin = compute_rope_cos_sin(head_dim, rope_theta, seq_len, device)

    # Forward then inverse
    x_rot = apply_rope(x, cos, sin)
    x_recovered = strip_rope(x_rot, cos, sin)

    max_err = (x - x_recovered).abs().max().item()
    print(f"RoPE roundtrip max error: {max_err:.2e}")
    assert max_err < 1e-5, f"RoPE roundtrip failed: max error {max_err}"
    print("PASS: RoPE roundtrip")


def test_rotate_half_shape():
    """rotate_half should preserve shape and swap halves with sign."""
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [1,1,1,4]
    result = rotate_half(x)
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.equal(result, expected), f"rotate_half wrong: {result}"
    print("PASS: rotate_half")


if __name__ == "__main__":
    test_rotate_half_shape()
    test_rope_roundtrip()
    print("All RoPE tests passed.")

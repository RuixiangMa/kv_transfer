"""Load models and extract KV cache from forward pass.

KV cache format (transformers 5.3.0 DynamicCache):
    cache.layers[i].keys   -> [batch, num_kv_heads, seq_len, head_dim]
    cache.layers[i].values -> [batch, num_kv_heads, seq_len, head_dim]
"""
import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_transfer.rope import compute_rope_cos_sin, strip_rope
from kv_transfer.config import (
    MODEL_8B, MODEL_14B,
    CALIBRATION_GLOB_PATTERNS, CALIBRATION_MAX_FILES,
)


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def load_model(model_path, device_map="auto"):
    """Load model in bf16 with cache enabled."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map
    )
    model.eval()
    return model


def extract_kv_cache(model, input_ids):
    """Forward pass, extract KV cache from all layers.

    Returns: dict {layer_idx: {"keys": K, "values": V}}
    K/V shape: [batch, num_kv_heads, seq_len, head_dim]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = outputs.past_key_values
    kv_dict = {}
    for layer_idx in range(len(cache.layers)):
        kv_dict[layer_idx] = {
            "keys": cache.layers[layer_idx].keys.clone(),
            "values": cache.layers[layer_idx].values.clone(),
        }
    return kv_dict


def strip_kv_rope(kv_dict, rope_theta, head_dim):
    """Strip RoPE from keys in-place. Values unchanged (no positional encoding).

    Args:
        kv_dict: {layer_idx: {"keys": K, "values": V}}
        rope_theta: float (1e6 for Qwen3)
        head_dim: int (128 for Qwen3)
    Returns: kv_dict with keys stripped of RoPE
    """
    # Infer seq_len from first layer's keys
    first_layer = kv_dict[0]
    keys = first_layer["keys"]
    seq_len = keys.shape[2]
    device = keys.device

    cos, sin = compute_rope_cos_sin(head_dim, rope_theta, seq_len, device)
    for layer_idx in kv_dict:
        k = kv_dict[layer_idx]["keys"]
        kv_dict[layer_idx]["keys"] = strip_rope(k, cos, sin)
    return kv_dict


def get_model_config(model):
    """Extract rope_theta and head_dim from model config."""
    config = model.config
    rope_theta = getattr(config, "rope_theta", 1000000)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    return rope_theta, head_dim, num_layers, num_kv_heads


def load_calibration_tokens(tokenizer, num_tokens=12000):
    """Read local text files for calibration data (instant, no generation needed)."""
    text_parts = []
    for pattern in CALIBRATION_GLOB_PATTERNS:
        for fpath in sorted(glob.glob(pattern, recursive=True))[:CALIBRATION_MAX_FILES]:
            try:
                with open(fpath, "r", errors="ignore") as f:
                    text_parts.append(f.read())
            except (IOError, OSError):
                pass
    text = chr(10).join(text_parts)
    print(f"Read {len(text_parts)} files, {len(text)} chars", flush=True)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    print(f"Tokenized to {ids.shape[0]} tokens", flush=True)
    return ids


def tokens_to_chunks(tokens, seq_len, num_seqs):
    """Split token sequence into fixed-length non-overlapping chunks."""
    num_available = tokens.shape[0] // seq_len
    num_seqs = min(num_seqs, num_available)
    chunks = [tokens[i*seq_len:(i+1)*seq_len] for i in range(num_seqs)]
    print(f"Prepared {len(chunks)} chunks of {seq_len} tokens", flush=True)
    return chunks


def extract_kv_chunks_to_cpu(model, chunks, device, rope_theta, head_dim):
    """Extract KV from multiple chunks, move to CPU immediately."""
    kv_rope = {}
    kv_stripped = {}
    seq_len = chunks[0].shape[0]
    cos, sin = compute_rope_cos_sin(head_dim, rope_theta, seq_len, "cpu")

    for i, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0).to(device)
        kv = extract_kv_cache(model, input_ids)
        for layer_idx in kv:
            k = kv[layer_idx]["keys"].cpu()
            v = kv[layer_idx]["values"].cpu()
            k_stripped = strip_rope(k, cos, sin)
            if i == 0:
                kv_rope[layer_idx] = {"keys": k, "values": v}
                kv_stripped[layer_idx] = {"keys": k_stripped, "values": v}
            else:
                kv_rope[layer_idx]["keys"] = torch.cat([kv_rope[layer_idx]["keys"], k], dim=2)
                kv_rope[layer_idx]["values"] = torch.cat([kv_rope[layer_idx]["values"], v], dim=2)
                kv_stripped[layer_idx]["keys"] = torch.cat([kv_stripped[layer_idx]["keys"], k_stripped], dim=2)
                kv_stripped[layer_idx]["values"] = torch.cat([kv_stripped[layer_idx]["values"], v], dim=2)
        del kv
        print(f"  Extracted chunk {i+1}/{len(chunks)}", flush=True)

    total_seq = kv_rope[0]["keys"].shape[2]
    print(f"Total calibration tokens: {total_seq}", flush=True)
    return kv_rope, kv_stripped

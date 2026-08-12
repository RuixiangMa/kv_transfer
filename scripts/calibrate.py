"""Full calibration pipeline: load models, extract KV, run R^2 analysis, fit mapper.

Optimized: reads local text files (no generation), reuses R^2 from Step 3, GPU ridge.
Usage: PYTHONUNBUFFERED=1 python3 -m scripts.calibrate [--num_seqs 100] [--seq_len 1024] [--k 8]
"""
import torch
import numpy as np
import os
import time
from kv_transfer.config import (
    MODEL_8B, MODEL_14B, OUTPUT_DIR, R2_RESULTS_PATH, MAPPER_PATH,
    DEFAULT_NUM_SEQS, DEFAULT_SEQ_LEN, DEFAULT_K, RIDGE_LAMBDA,
    FIT_DEVICE, SOURCE_DEVICE,
)
from kv_transfer.kv_cache import (
    load_model, load_tokenizer, get_model_config,
    load_calibration_tokens, tokens_to_chunks, extract_kv_chunks_to_cpu,
)
from kv_transfer.analysis import analyze_linear_structure
from kv_transfer.mapper import select_topk_sources_from_r2, fit_mapper, save_mapper


def main(num_seqs=DEFAULT_NUM_SEQS, seq_len=DEFAULT_SEQ_LEN, k=DEFAULT_K):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load 8B ---
    print("=" * 60, flush=True)
    print("Step 1: Load Qwen3-8B and extract calibration KV", flush=True)
    print("=" * 60, flush=True)
    tok = load_tokenizer(MODEL_8B)
    model_8b = load_model(MODEL_8B, device_map=SOURCE_DEVICE)
    rope_theta_8b, head_dim_8b, num_layers_8b, num_kv_8b = get_model_config(model_8b)
    print(f"8B: {num_layers_8b} layers, {num_kv_8b} KV heads, head_dim={head_dim_8b}", flush=True)

    chunks = load_calibration_tokens(model_8b, tok, num_tokens=num_seqs * seq_len + 500, device=SOURCE_DEVICE)
    chunks = tokens_to_chunks(chunks, seq_len, num_seqs)

    source_kv_rope, source_kv_stripped = extract_kv_chunks_to_cpu(
        model_8b, chunks, SOURCE_DEVICE, rope_theta_8b, head_dim_8b
    )

    del model_8b
    torch.cuda.empty_cache()
    print("8B unloaded.", flush=True)

    # --- Load 14B ---
    print("\n" + "=" * 60, flush=True)
    print("Step 2: Load Qwen3-14B and extract calibration KV", flush=True)
    print("=" * 60, flush=True)
    model_14b = load_model(MODEL_14B, device_map="auto")
    rope_theta_14b, head_dim_14b, num_layers_14b, num_kv_14b = get_model_config(model_14b)
    print(f"14B: {num_layers_14b} layers, {num_kv_14b} KV heads, head_dim={head_dim_14b}", flush=True)

    target_device = next(model_14b.parameters()).device
    target_kv_rope, target_kv_stripped = extract_kv_chunks_to_cpu(
        model_14b, chunks, target_device, rope_theta_14b, head_dim_14b
    )

    del model_14b
    torch.cuda.empty_cache()
    print("14B unloaded.", flush=True)

    # --- R^2 Analysis ---
    print("\n" + "=" * 60, flush=True)
    print("Step 3: R^2 Linear Structure Analysis", flush=True)
    print("=" * 60, flush=True)
    for kv in [source_kv_stripped, target_kv_stripped, source_kv_rope, target_kv_rope]:
        for layer_idx in kv:
            kv[layer_idx]["keys"] = kv[layer_idx]["keys"].cpu()
            kv[layer_idx]["values"] = kv[layer_idx]["values"].cpu()

    r2_results = analyze_linear_structure(
        source_kv_stripped, target_kv_stripped,
        source_kv_rope, target_kv_rope,
        num_layers_8b, num_layers_14b, num_kv_8b
    )
    np.savez(R2_RESULTS_PATH,
             K_rope=r2_results["K_rope"], K_stripped=r2_results["K_stripped"], V=r2_results["V"])
    print(f"R^2 results saved to {R2_RESULTS_PATH}", flush=True)

    # --- Top-k Selection + Ridge Fit ---
    print("\n" + "=" * 60, flush=True)
    print(f"Step 4: Top-{k} Source Selection + Ridge Mapper Fitting", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    selected = select_topk_sources_from_r2(
        r2_results["K_stripped"], r2_results["V"], k
    )
    print(f"Top-k selection done in {time.time()-t0:.1f}s (reused R^2 from Step 3)", flush=True)
    for l_t in [0, num_layers_14b // 2, num_layers_14b - 1]:
        print(f"  Target layer {l_t}: sources {selected[l_t]}", flush=True)

    t0 = time.time()
    mapper = fit_mapper(
        source_kv_stripped, target_kv_stripped, selected,
        num_layers_14b, num_kv_8b, lam=RIDGE_LAMBDA, device=FIT_DEVICE
    )
    print(f"Ridge fitting done in {time.time()-t0:.1f}s (GPU + cached X^T X)", flush=True)

    save_mapper(mapper, MAPPER_PATH)
    print(f"\nCalibration complete. Output in {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seqs", type=int, default=DEFAULT_NUM_SEQS)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    main(args.num_seqs, args.seq_len, args.k)

"""R^2 linear structure analysis - optimized with normal equations + GPU."""
import torch
import numpy as np


def compute_r2_single_source(X, Y):
    """Fit single-source OLS and compute R^2 using normal equations (faster than lstsq)."""
    X = X.float()
    Y = Y.float()
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    # Normal equations: W = (X_c^T X_c)^{-1} X_c^T Y_c
    XtX = X_c.T @ X_c  # [d, d]
    XtY = X_c.T @ Y_c  # [d, d]
    W = torch.linalg.solve(XtX, XtY)
    Y_pred = X_c @ W
    ss_res = ((Y_c - Y_pred) ** 2).sum()
    ss_tot = (Y_c ** 2).sum()
    if ss_tot < 1e-10:
        return 0.0
    return (1.0 - (ss_res / ss_tot).item())


def compute_r2_matrix(source_kv, target_kv, num_source_layers, num_target_layers,
                      num_kv_heads, cache_type="K_stripped",):
    """Compute head-averaged R^2 matrix [num_source_layers, num_target_layers]."""
    r2_matrix = np.zeros((num_source_layers, num_target_layers))
    for l_t in range(num_target_layers):
        for l_s in range(num_source_layers):
            r2_values = []
            for h in range(num_kv_heads):
                if cache_type == "V":
                    X = source_kv[l_s]["values"][0, h, :, :]
                    Y = target_kv[l_t]["values"][0, h, :, :]
                else:
                    X = source_kv[l_s]["keys"][0, h, :, :]
                    Y = target_kv[l_t]["keys"][0, h, :, :]
                r2_values.append(compute_r2_single_source(X, Y))
            r2_matrix[l_s, l_t] = np.mean(r2_values)
    return r2_matrix


def analyze_linear_structure(source_kv_stripped, target_kv_stripped,
                              source_kv_rope, target_kv_rope,
                              num_source_layers, num_target_layers, num_kv_heads):
    """Run full R^2 analysis for all three cache types. Returns dict with R^2 matrices."""
    results = {}
    for cache_type in ["K_rope", "K_stripped", "V"]:
        src = source_kv_rope if cache_type == "K_rope" else source_kv_stripped
        tgt = target_kv_rope if cache_type == "K_rope" else target_kv_stripped
        r2 = compute_r2_matrix(src, tgt, num_source_layers, num_target_layers,
                               num_kv_heads, cache_type)
        results[cache_type] = r2
        print(f"\n{cache_type} R^2 matrix [{num_source_layers}x{num_target_layers}]:", flush=True)
        print(f"  Max: {r2.max():.4f}", flush=True)
        print(f"  Diagonal mean: {np.mean([r2[min(i, num_source_layers-1), i] for i in range(num_target_layers)]):.4f}", flush=True)
        print(f"  Per-target best source layer mean: {r2.max(axis=0).mean():.4f}", flush=True)

    k_stripped_best = results["K_stripped"].max(axis=0).mean()
    v_best = results["V"].max(axis=0).mean()
    k_rope_best = results["K_rope"].max(axis=0).mean()
    print(f"\n--- Summary ---", flush=True)
    print(f"K_stripped best-per-target mean R^2: {k_stripped_best:.4f}", flush=True)
    print(f"K_rope best-per-target mean R^2: {k_rope_best:.4f}", flush=True)
    print(f"V best-per-target mean R^2: {v_best:.4f}", flush=True)
    print(f"K vs V gap: {k_stripped_best - v_best:.4f}", flush=True)
    print(f"RoPE effect (K_stripped - K_rope): {k_stripped_best - k_rope_best:.4f}", flush=True)
    return results

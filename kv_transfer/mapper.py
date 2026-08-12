"""Ridge mapper: top-k source selection + closed-form ridge regression.

Optimized: reuses R^2 from Step 3 (no recomputation), caches X^T X per layer, uses GPU.
"""
import torch
import numpy as np
from kv_transfer.rope import apply_rope


def select_topk_sources_from_r2(r2_k_stripped, r2_v, k):
    """Select top-k source layers per target layer using pre-computed R^2 matrices.

    Args:
        r2_k_stripped: [num_source_layers, num_target_layers] R^2 matrix for K_stripped
        r2_v: [num_source_layers, num_target_layers] R^2 matrix for V
        k: number of source layers to select
    Returns: dict {target_layer_idx: [source_layer_indices sorted ascending]}
    """
    r2_avg = (r2_k_stripped + r2_v) / 2.0
    selected = {}
    for l_t in range(r2_avg.shape[1]):
        scores = r2_avg[:, l_t]
        topk_idx = np.argsort(scores)[::-1][:k]
        selected[l_t] = sorted(topk_idx.tolist())
    return selected


def fit_mapper(source_kv_stripped, target_kv_stripped, selected_sources,
               num_target_layers, num_kv_heads, lam=0.01, device="cuda:7"):
    """Fit per-head ridge mapper. Caches X^T X per layer, uses GPU."""
    mapper = {}
    for l_t in range(num_target_layers):
        src_indices = selected_sources[l_t]

        # Build feature matrices on GPU
        X_K_list = []
        X_V_list = []
        for l_s in src_indices:
            sk = source_kv_stripped[l_s]["keys"][0].to(device).float()
            sv = source_kv_stripped[l_s]["values"][0].to(device).float()
            seq_len = sk.shape[1]
            X_K_list.append(sk.permute(1, 0, 2).reshape(seq_len, -1))
            X_V_list.append(sv.permute(1, 0, 2).reshape(seq_len, -1))
        X_K = torch.cat(X_K_list, dim=1)
        X_V = torch.cat(X_V_list, dim=1)

        # Cache X^T X + LU factorize once per layer
        X_K_mean = X_K.mean(0)
        X_K_c = X_K - X_K_mean
        XtX_K = X_K_c.T @ X_K_c + lam * torch.eye(X_K_c.shape[1], device=device)
        lu_K, pivots_K = torch.linalg.lu_factor(XtX_K)

        X_V_mean = X_V.mean(0)
        X_V_c = X_V - X_V_mean
        XtX_V = X_V_c.T @ X_V_c + lam * torch.eye(X_V_c.shape[1], device=device)
        lu_V, pivots_V = torch.linalg.lu_factor(XtX_V)

        # Fit per head (only X^T Y + solve using cached LU)
        W_K_list = []
        b_K_list = []
        W_V_list = []
        b_V_list = []
        for h in range(num_kv_heads):
            Y_K = target_kv_stripped[l_t]["keys"][0, h, :, :].to(device).float()
            Y_V = target_kv_stripped[l_t]["values"][0, h, :, :].to(device).float()

            Y_K_mean = Y_K.mean(0)
            Y_K_c = Y_K - Y_K_mean
            XtY_K = X_K_c.T @ Y_K_c
            W_k = torch.linalg.lu_solve(lu_K, pivots_K, XtY_K)
            b_k = Y_K_mean - X_K_mean @ W_k

            Y_V_mean = Y_V.mean(0)
            Y_V_c = Y_V - Y_V_mean
            XtY_V = X_V_c.T @ Y_V_c
            W_v = torch.linalg.lu_solve(lu_V, pivots_V, XtY_V)
            b_v = Y_V_mean - X_V_mean @ W_v

            W_K_list.append(W_k.cpu())
            b_K_list.append(b_k.cpu())
            W_V_list.append(W_v.cpu())
            b_V_list.append(b_v.cpu())

        mapper[l_t] = {
            "selected_sources": src_indices,
            "W_K": torch.stack(W_K_list),
            "b_K": torch.stack(b_K_list),
            "W_V": torch.stack(W_V_list),
            "b_V": torch.stack(b_V_list),
        }

        if l_t % 10 == 0:
            print(f"  Fitted layer {l_t}/{num_target_layers}", flush=True)

        del X_K, X_V, X_K_c, X_V_c, XtX_K, XtX_V, lu_K, lu_V
        torch.cuda.empty_cache()

    return mapper


def apply_mapper(source_kv_stripped, mapper, num_target_layers, num_kv_heads,
                 target_rope_cos=None, target_rope_sin=None):
    """Apply mapper to source KV, produce mapped KV for target."""
    mapped_kv = {}
    for l_t in range(num_target_layers):
        m = mapper[l_t]
        src_indices = m["selected_sources"]

        X_K_list = []
        X_V_list = []
        for l_s in src_indices:
            sk = source_kv_stripped[l_s]["keys"][0]
            sv = source_kv_stripped[l_s]["values"][0]
            seq_len = sk.shape[1]
            X_K_list.append(sk.permute(1, 0, 2).reshape(seq_len, -1))
            X_V_list.append(sv.permute(1, 0, 2).reshape(seq_len, -1))
        X_K = torch.cat(X_K_list, dim=1).float()
        X_V = torch.cat(X_V_list, dim=1).float()

        W_K = m["W_K"].float().to(X_K.device)
        b_K = m["b_K"].float().to(X_K.device)
        W_V = m["W_V"].float().to(X_V.device)
        b_V = m["b_V"].float().to(X_V.device)

        mapped_K_list = [X_K @ W_K[h] + b_K[h] for h in range(num_kv_heads)]
        mapped_K = torch.stack(mapped_K_list)
        mapped_V_list = [X_V @ W_V[h] + b_V[h] for h in range(num_kv_heads)]
        mapped_V = torch.stack(mapped_V_list)

        if target_rope_cos is not None:
            mapped_K = mapped_K.unsqueeze(0)
            mapped_K = apply_rope(mapped_K, target_rope_cos, target_rope_sin)
        else:
            mapped_K = mapped_K.unsqueeze(0)

        mapped_V = mapped_V.unsqueeze(0)
        mapped_kv[l_t] = {
            "keys": mapped_K.bfloat16(),
            "values": mapped_V.bfloat16(),
        }

    return mapped_kv


def save_mapper(mapper, path):
    torch.save(mapper, path)
    print(f"Mapper saved to {path}")


def load_mapper(path):
    return torch.load(path, weights_only=False)

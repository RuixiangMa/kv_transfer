# Cross-Model KV Cache Transfer

Closed-form linear mapping for reusing KV cache across LLM family members, skipping re-prefill.

Implements [arXiv:2608.03893](https://arxiv.org/abs/2608.03893) (NVIDIA).

## Usage

```bash
# Calibrate mapper
PYTHONUNBUFFERED=1 python3 -m scripts.calibrate --num_seqs 100 --seq_len 1024 --k 8

# Evaluate (attention cosine + HellaSwag retention)
python3 -m scripts.evaluate
```

## Structure

```
kv_transfer/
├── kv_transfer/       # Core package
│   ├── config.py      # Model paths, hyperparameters
│   ├── rope.py        # RoPE strip/re-apply
│   ├── kv_cache.py    # Model loading + KV extraction
│   ├── analysis.py    # R² linear structure analysis
│   ├── mapper.py      # Ridge mapper (fit + apply + top-k)
│   └── evaluation.py  # HellaSwag + attention cosine
├── scripts/           # CLI entry points
│   ├── calibrate.py
│   └── evaluate.py
└── tests/
```

## Requirements

- torch 2.9+, transformers 5.3+, numpy

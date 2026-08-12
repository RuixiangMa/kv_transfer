#!/bin/bash
set -e

cd "$(dirname "$0")"

# Step 1: Calibrate mapper
PYTHONUNBUFFERED=1 python3 -m scripts.calibrate --num_seqs 100 --seq_len 1024 --k 8

# Step 2: Evaluate (attention cosine + HellaSwag retention)
python3 -m scripts.evaluate

"""Central configuration: model paths, hyperparameters, device config.

All constants live here. Model-specific values (rope_theta, head_dim,
num_layers) are defaults overridden at runtime by get_model_config().
"""
import os

# Model paths
MODEL_8B = "/cache/models/Qwen3-8B"
MODEL_14B = "/cache/models/Qwen3-14B"

# RoPE / model config (Qwen3 defaults, overridden by get_model_config at runtime)
ROPE_THETA = 1000000
HEAD_DIM = 128

# Calibration hyperparameters
DEFAULT_NUM_SEQS = 100
DEFAULT_SEQ_LEN = 1024
DEFAULT_K = 8
RIDGE_LAMBDA = 0.01
FIT_DEVICE = "cuda:7"

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_14b")
MAPPER_PATH = os.path.join(OUTPUT_DIR, "mapper.pt")
R2_RESULTS_PATH = os.path.join(OUTPUT_DIR, "r2_results.npz")

# Calibration data source
CALIBRATION_GLOB_PATTERNS = [
    "/cache/vllm-omni-pr5840/**/*.py",
    "/cache/vllm-omni-pr5840/**/*.md",
]
CALIBRATION_MAX_FILES = 200

# Device
SOURCE_DEVICE = "cuda:0"

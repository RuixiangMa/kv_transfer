"""Evaluation CLI: run attention cosine + HellaSwag retention.

Usage: python3 -m scripts.evaluate
"""
import torch
from kv_transfer.config import MODEL_8B, MODEL_14B, MAPPER_PATH, SOURCE_DEVICE
from kv_transfer.kv_cache import load_model, load_tokenizer
from kv_transfer.mapper import load_mapper
from kv_transfer.evaluation import run_attention_cosine_eval, run_hellaswag_eval


def main():
    mapper = load_mapper(MAPPER_PATH)
    tok = load_tokenizer(MODEL_8B)
    model_8b = load_model(MODEL_8B, device_map=SOURCE_DEVICE)
    model_14b = load_model(MODEL_14B, device_map="auto")

    run_attention_cosine_eval(model_8b, model_14b, tok, mapper)
    run_hellaswag_eval(model_8b, model_14b, tok, mapper)

    del model_8b, model_14b
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

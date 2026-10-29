"""Plan 13: INT8 and 4-bit checkpoints of the served arm, with llm-compressor.

Runs in `.venv-serve` only. Calibration prompts come from folds 2/3/4, so no
query the ladder reports on shapes the quantisation.

  w8a8   SmoothQuant then GPTQ, INT8 weights and activations
  w4a16  GPTQ, 4-bit weights, 16-bit activations

FP8 is not offered: Ampere cards (the RTX 3080, and the RTX A6000 the ladder
ran on) have no FP8 tensor cores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cross_encoder import check_training_folds
from src.ranker import TRAIN_FOLDS

SCHEMES: tuple[str, ...] = ("w8a8", "w4a16")
CALIBRATION_ROWS = 512


def calibration_rows(matrix, n: int = CALIBRATION_ROWS, seed: int = 0):
    rows = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)]
    if len(rows) < n:
        raise ValueError(f"{n} calibration rows asked for, {len(rows)} in the training folds")
    sample = rows.sample(n=n, random_state=seed)
    check_training_folds(sample)
    return sample


def recipe(scheme: str) -> list:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    from llmcompressor.modifiers.quantization import GPTQModifier

    if scheme == "w8a8":
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        return [SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
    return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]


def _main() -> int:
    import pandas as pd
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    from src.cross_encoder import text_maps
    from src.llm_list import encode_prompt, list_training_set
    from src.llm_point import DEFAULT_MAX_LENGTH, encode_pair

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--kind", choices=["point", "list"], required=True)
    parser.add_argument("--scheme", choices=list(SCHEMES), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    matrix = pd.read_parquet("data/features/train.parquet",
                             columns=["query_id", "product_id", "gain", "fold"])
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)]
    query_text, doc_text = text_maps(matrix, Path("data/combined/products.parquet"),
                                     Path("data/combined/judgements.parquet"))
    query_text = {str(k): v for k, v in query_text.items()}
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if args.kind == "point":
        rows = calibration_rows(matrix)
        ids = [encode_pair(tok, query_text[str(q)], doc_text[p], args.max_length)[0]
               for q, p in zip(rows["query_id"], rows["product_id"])]
        max_seq = args.max_length
    else:
        examples = list_training_set(query_text, doc_text)[:CALIBRATION_ROWS]
        ids = [encode_prompt(tok, q, d) for q, d, _ in examples]
        max_seq = max(len(x) for x in ids)
    dataset = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
    oneshot(model=args.model_dir, dataset=dataset, recipe=recipe(args.scheme),
            max_seq_length=max_seq, num_calibration_samples=len(ids), output_dir=args.out)
    tok.save_pretrained(args.out)
    print(f"{args.scheme} checkpoint -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

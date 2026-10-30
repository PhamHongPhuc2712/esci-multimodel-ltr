"""Build the Plan 13 evaluation bundle: the data `src.llm_rank_report` reads, plus the two bge arms.

The file list was taken by tracing every open the report's inputs make on the
laptop (fold 0, test, the fold-1 warm-up and the published references). Only
the product table is slimmed: 1.8 GB whole, it is cut to the judged products
of both splits with the three columns `src.cross_encoder.text_maps` reads.
The trained LLMs are already on the server and do not go up.
"""
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "build/cloud/eval-stage"
OUT = ROOT / "build/cloud/plan13-eval-bundle.tar"

COPIED = [
    "data/combined/judgements.parquet",
    "data/features/train.parquet",
    "data/features/test.parquet",
    "data/features/stage2-train.parquet",
    "data/features/stage2-test.parquet",
    "data/features/stage-signals-fold0.parquet",
    "data/features/stage-signals-test.parquet",
    "docs/results/fine-rank.json",
    "docs/results/fine-rank-test.json",
    "docs/results/fine-rank-test-full.json",
    "docs/results/size.json",
    "docs/results/size-test.json",
]
MODELS = ["models/cross-encoder/bge-groups/lambda", "models/cross-encoder/bge-large-groups/lambda"]

(STAGE / "data/combined").mkdir(parents=True, exist_ok=True)
judged = set()
for split in ("train", "test"):
    judged |= set(pd.read_parquet(ROOT / f"data/features/{split}.parquet", columns=["product_id"])["product_id"])
products = pd.read_parquet(ROOT / "data/combined/products.parquet", columns=["product_id", "product_title", "description"])
products = products.loc[products["product_id"].isin(judged)]
products.to_parquet(STAGE / "data/combined/products.parquet", index=False)

subprocess.run(["git", "archive", "--format=tar", "-o", str(STAGE / "code.tar"), "HEAD", "src", "pyproject.toml"],
               cwd=ROOT, check=True)

# Plain tar: the models are already-compressed safetensors and gzip buys nothing.
with tarfile.open(OUT, "w") as tar:
    tar.add(STAGE / "data/combined/products.parquet", arcname="data/combined/products.parquet")
    tar.add(STAGE / "code.tar", arcname="code.tar")
    for rel in COPIED + MODELS:
        tar.add(ROOT / rel, arcname=rel)

print(f"{len(products):,} judged products")
print(f"{OUT} {OUT.stat().st_size / 2**20:.1f} MB")

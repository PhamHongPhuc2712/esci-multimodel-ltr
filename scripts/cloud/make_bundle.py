"""Build the Plan 13 training bundle for a rented GPU: code plus only the data training reads.

The full product table is 1.8 GB; training needs titles and descriptions of
training-fold products only, so the bundle carries a slim table with the same
columns `src.cross_encoder.text_maps` reads. Nothing from fold 0, fold 1 or
the test split goes up except the out-of-fold Stage 2 scores Plan 8 made.
"""
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "build/cloud/stage"
OUT = ROOT / "build/cloud/plan13-bundle.tar.gz"
TRAIN_FOLDS = (2, 3, 4)

(STAGE / "data/features").mkdir(parents=True, exist_ok=True)
(STAGE / "data/combined").mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(ROOT / "data/features/train.parquet", columns=["query_id", "product_id", "gain", "fold"])
train = train.loc[train["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
train.to_parquet(STAGE / "data/features/train.parquet", index=False)

oof = pd.read_parquet(ROOT / "data/features/stage2-train-oof.parquet")
oof = oof.merge(train[["query_id", "product_id"]], on=["query_id", "product_id"])
oof.to_parquet(STAGE / "data/features/stage2-train-oof.parquet", index=False)

products = pd.read_parquet(ROOT / "data/combined/products.parquet", columns=["product_id", "product_title", "description"])
products = products.loc[products["product_id"].isin(set(train["product_id"]))]
products.to_parquet(STAGE / "data/combined/products.parquet", index=False)

judgements = pd.read_parquet(ROOT / "data/combined/judgements.parquet", columns=["query_id", "query"])
judgements = judgements.loc[judgements["query_id"].isin(set(train["query_id"]))].drop_duplicates("query_id")
judgements.to_parquet(STAGE / "data/combined/judgements.parquet", index=False)

# The code exactly as committed, so the cloud runs what the repo records.
subprocess.run(["git", "archive", "--format=tar", "-o", str(STAGE / "code.tar"), "HEAD", "src", "pyproject.toml"],
               cwd=ROOT, check=True)

with tarfile.open(OUT, "w:gz") as tar:
    tar.add(STAGE / "data", arcname="data")
    tar.add(STAGE / "code.tar", arcname="code.tar")
    tar.add(ROOT / "scripts/cloud/setup_and_train.sh", arcname="setup_and_train.sh")

print(f"{len(train):,} training rows, {len(products):,} products, {len(judgements):,} queries")
print(f"{OUT} {OUT.stat().st_size / 2**20:.1f} MB")

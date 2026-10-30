#!/usr/bin/env bash
# Plan 13 fold-0 evaluation on the rented GPU, after every training run ends.
#
#   nohup bash eval_fold0.sh > logs/eval.out 2>&1 &
#
# Waits for the listwise retry, unpacks the evaluation bundle over the repo
# (its full train.parquet replaces the training bundle's folds-2/3/4 slice),
# installs what the report imports without moving torch, transformers or peft,
# then runs esci_rerank.rerank.llm_rank_report on fold 0. The test split is not touched here.
set -uo pipefail

WORK=/root/plan13
REPO=$WORK/repo
LOGS=$WORK/logs
cd "$WORK"

while ! grep -q "RETRY DONE" "$LOGS/run.log" 2>/dev/null; do sleep 60; done
echo "eval: starting  ($(date))" >> "$LOGS/run.log"

tar xf plan13-eval-bundle.tar -C "$REPO" || { echo "eval: failed to unpack" >> "$LOGS/run.log"; exit 1; }
tar xf "$REPO/code.tar" -C "$REPO" && rm "$REPO/code.tar"

python3 -m pip freeze | grep -iE '^(torch|transformers|peft)==' > constraints.txt
python3 -m pip install --quiet -c constraints.txt \
    "sentence-transformers==6.1.0" "lightgbm==4.7.0" scipy zstandard \
    "datasets>=3.0" "accelerate>=1.1.0" pytrec-eval-terrier 2>&1 | tail -3

cd "$REPO"
export PYTHONPATH="$REPO" HF_HUB_DISABLE_PROGRESS_BARS=1
python3 -c "import esci_rerank.rerank.llm_rank_report as r, esci_rerank.rerank.distill_report, esci_rerank.blend.blend_report, esci_rerank.blend.stage_signals, esci_rerank.rerank.cross_encoder, esci_rerank.ranking.rank_report; print('imports ok', r.available_arms())" \
    >> "$LOGS/eval-fold0.log" 2>&1 || { echo "eval: failed imports; see eval-fold0.log" >> "$LOGS/run.log"; exit 1; }

python3 -u -m esci_rerank.rerank.llm_rank_report >> "$LOGS/eval-fold0.log" 2>&1
rc=$?
echo "eval: exit $rc  ($(date))" >> "$LOGS/run.log"
echo "EVAL DONE $(date)" >> "$LOGS/run.log"

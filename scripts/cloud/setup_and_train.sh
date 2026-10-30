#!/usr/bin/env bash
# Plan 13 on a rented GPU: unpack, install, probe, train, pack.
#
#   bash setup_and_train.sh            # trains listwise, pointwise, then 4B (default)
#   ARMS="4b" bash setup_and_train.sh  # or any subset
#
# Run it under nohup or tmux: an SSH drop must not kill a 5-hour run.
# Every arm keeps the laptop recipe's examples per optimiser update (16);
# only the split between batch and accumulation and the gradient
# checkpointing change, and the training record says so.
set -uo pipefail

ARMS="${ARMS:-list point 4b}"
WORK=/root/plan13
REPO=$WORK/repo
OUT=$WORK/out
LOGS=$WORK/logs
mkdir -p "$REPO" "$OUT" "$LOGS"
cd "$WORK"

if [ ! -d "$REPO/esci_rerank" ]; then
    tar xzf plan13-bundle.tar.gz
    tar xf code.tar -C "$REPO"
    mv data "$REPO/data"
fi

# PyTorch routes some small ops to Triton kernels, and Triton compiles a
# launcher with a C compiler and the Python headers at first use. Some
# images ship neither.
if ! command -v gcc >/dev/null 2>&1 || ! python3 -c "import sysconfig,os,sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc python3-dev >/dev/null \
        || { echo "could not install gcc and python3-dev"; exit 1; }
fi

# Images differ: some ship only python3. Everything goes through $PY.
PY="${PY:-$(command -v python3 || command -v python)}"
[ -n "$PY" ] || { echo "no python on this image"; exit 1; }

"$PY" - <<'PYEOF'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__,
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name())
PYEOF
"$PY" -m pip install --quiet \
    "transformers==5.17.0" "peft==0.21.2" pandas pyarrow numpy 2>&1 | tail -2
"$PY" -c "import transformers, peft; print('transformers', transformers.__version__, 'peft', peft.__version__)" \
    || { echo "dependencies did not install"; exit 1; }

cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HUB_DISABLE_PROGRESS_BARS=1

# One arm: probe batch layouts largest first (16 examples an update each),
# without checkpointing; fall back to checkpointing if none fits.
train_arm() {
    local name=$1 module=$2 out=$3; shift 3
    local extra=("$@") chosen=""
    for layout in "16 1" "8 2" "4 4"; do
        read -r b a <<<"$layout"
        local log="$LOGS/probe-$name-b${b}x${a}.log"
        if "$PY" -m "$module" train --out "$out" "${extra[@]}" --batch-size "$b" --grad-accum "$a" \
                --no-gradient-checkpointing --probe 20 > "$log" 2>&1; then
            chosen="--batch-size $b --grad-accum $a --no-gradient-checkpointing"
            grep -E '"(peak_reserved_gb|projected_epoch_hours)"' "$log"
            break
        fi
        # Only running out of memory means "try a smaller batch". Anything
        # else is a real failure, and guessing past it wastes paid hours.
        if ! grep -q -E 'OutOfMemoryError|CUDA out of memory|Failed to create GPU mapping' "$log"; then
            echo "$name: probe failed for a reason other than memory; see $log"
            tail -5 "$log"
            exit 1
        fi
        echo "$name: batch $b x $a did not fit without checkpointing"
    done
    if [ -z "$chosen" ]; then
        chosen="--batch-size 4 --grad-accum 4"
        echo "$name: falling back to gradient checkpointing"
    fi
    echo "$name: training with $chosen  ($(date))"
    # shellcheck disable=SC2086
    "$PY" -u -m "$module" train --out "$out" "${extra[@]}" $chosen > "$LOGS/train-$name.log" 2>&1
    local rc=$?
    echo "$name: exit $rc  ($(date))" | tee -a "$LOGS/train-$name.log"
    if [ $rc -eq 0 ]; then
        tar cf "$OUT/$name.tar" -C "$(dirname "$out")" "$(basename "$out")"
        sha256sum "$OUT/$name.tar" > "$OUT/$name.tar.sha256"
        echo "$name: packed $OUT/$name.tar ($(du -h "$OUT/$name.tar" | cut -f1))"
    fi
}

for arm in $ARMS; do
    case $arm in
        point) train_arm point esci_rerank.rerank.llm_point models/llm-rank/point-0.6b ;;
        list)  train_arm list  esci_rerank.rerank.llm_list  models/llm-rank/list-1.7b ;;
        4b)    train_arm 4b    esci_rerank.rerank.llm_point models/llm-rank/point-4b \
                   --backbone Qwen/Qwen3-Reranker-4B --lora-rank 16 ;;
        *)     echo "unknown arm $arm" ;;
    esac
done
echo "ALL DONE $(date)" | tee "$LOGS/done"

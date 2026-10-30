#!/usr/bin/env bash
# Plan 13 Phases 3 and 4 (GPU part) on the rented A6000, for the arm fold 0 chose.
#
#   nohup bash phase3.sh > logs/phase3.out 2>&1 &
#
# Runs after setup_phase3.sh has built both environments. Then: the two in-process
# rungs, the two quantised checkpoints, the four served rungs (a fresh server
# each), the load test on the deployed rung, and the reference and deployed
# rungs on the test split. A rung that cannot be built is logged and left out;
# a served bf16 rung that disagrees with the reference stops the ladder.
set -uo pipefail

WORK=/root/plan13
REPO=$WORK/repo
LOGS=$WORK/logs
RUN=$LOGS/run.log
MODEL=models/llm-rank/point-4b
KIND=point
NAME=point
URL=http://127.0.0.1:8000
BASE_FLAGS="--max-model-len 512 --gpu-memory-utilization 0.85 --port 8000"
PRICE=0.40
PRICE_SOURCE="gpu.ai on-demand RTX A6000, Canada, \$0.40/hr as shown in the launch console on 2026-10-30"

cd "$REPO"
export PYTHONPATH="$REPO" HF_HUB_DISABLE_PROGRESS_BARS=1
# FlashInfer's top-k/top-p sampler JIT-compiles at startup and this image has
# no ninja or nvcc. Every request here is greedy (temperature 0), so the
# PyTorch sampler gives the same tokens. Recorded in every rung's flags.
export VLLM_USE_FLASHINFER_SAMPLER=0
SERVE_ENV="VLLM_USE_FLASHINFER_SAMPLER=0"
log() { echo "$*  ($(date))" >> "$RUN"; }
# A rerun resumes: rungs already in llm-speed.json and checkpoints already
# on disk are kept, not measured or built twice.
measured() { python3 -c "import json,sys; sys.exit(0 if '$1' in json.load(open('docs/results/llm-speed.json'))['rungs'] else 1)" 2>/dev/null; }

while ! grep -q "QUANT ENV READY" "$LOGS/build-quant.log" 2>/dev/null; do
    grep -q "^error\|Error" "$LOGS/build-quant.log" 2>/dev/null && { log "phase3: failed, quantisation env did not build"; exit 1; }
    sleep 30
done
log "phase3: starting"

speed() {  # rung, extra args...
    local rung=$1; shift
    python3 -u -m src.llm_speed_report --rung "$rung" --kind $KIND --model-dir $MODEL "$@" \
        >> "$LOGS/speed-$rung.log" 2>&1
}

server_up() {  # model dir, flags
    local dir=$1 flags=$2
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && sleep 20
    setsid .venv-serve/bin/vllm serve "$dir" --served-model-name $NAME $flags \
        > "$LOGS/vllm-$(basename "$dir")-$(date +%s).log" 2>&1 < /dev/null &
    SERVER=$!
    for _ in $(seq 180); do
        curl -sf "$URL/health" >/dev/null && return 0
        kill -0 $SERVER 2>/dev/null || return 1
        sleep 5
    done
    return 1
}

server_down() {
    kill -- -"$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
    for _ in $(seq 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
}

# 1. In-process rungs, no server up.
for rung in hf_bf16_batched hf_bf16_loop; do
    measured $rung && { log "phase3: $rung already measured, kept"; continue; }
    speed $rung && log "phase3: $rung exit 0" || { log "phase3: $rung failed"; exit 1; }
done

# 2. Quantised checkpoints, calibrated on folds 2/3/4.
for scheme in w8a8 w4a16; do
    [ -f "$MODEL-$scheme/config.json" ] && { log "phase3: quantize $scheme already built, kept"; continue; }
    .venv-quant/bin/python -u -m src.quantize_llm --model-dir $MODEL --kind $KIND --scheme $scheme \
        --out $MODEL-$scheme > "$LOGS/quantize-$scheme.log" 2>&1 \
        && log "phase3: quantize $scheme exit 0" || log "phase3: quantize $scheme failed; rung left out"
done

# 3. Served rungs, a fresh server each.
declare -A DIR FLAGS
DIR[vllm_bf16]=$MODEL;             FLAGS[vllm_bf16]="--dtype bfloat16 $BASE_FLAGS --no-enable-prefix-caching"
DIR[vllm_bf16_prefix]=$MODEL;      FLAGS[vllm_bf16_prefix]="--dtype bfloat16 $BASE_FLAGS --enable-prefix-caching"
DIR[vllm_w8a8_prefix]=$MODEL-w8a8;   FLAGS[vllm_w8a8_prefix]="$BASE_FLAGS --enable-prefix-caching"
DIR[vllm_w4a16_prefix]=$MODEL-w4a16; FLAGS[vllm_w4a16_prefix]="$BASE_FLAGS --enable-prefix-caching"
for rung in vllm_bf16 vllm_bf16_prefix vllm_w8a8_prefix vllm_w4a16_prefix; do
    dir=${DIR[$rung]}
    measured $rung && { log "phase3: $rung already measured, kept"; continue; }
    [ -f "$dir/config.json" ] || { log "phase3: $rung skipped, no checkpoint at $dir"; continue; }
    if ! server_up "$dir" "${FLAGS[$rung]}"; then
        log "phase3: $rung failed, server did not come up"; server_down
        [ $rung = vllm_bf16 ] && exit 1 || continue
    fi
    speed $rung --served-name $NAME --server-flags "$SERVE_ENV vllm serve $dir --served-model-name $NAME ${FLAGS[$rung]}"
    rc=$?
    server_down
    log "phase3: $rung exit $rc"
    [ $rung = vllm_bf16 ] && [ $rc -ne 0 ] && { log "phase3: served bf16 disagrees or failed; ladder stopped"; exit 1; }
done

# 4. Load test on the deployed rung, fresh server.
deployed=$(python3 -c "import json; print(json.load(open('docs/results/llm-speed.json'))['deployed_rung'])")
dir=${DIR[$deployed]}; flags=${FLAGS[$deployed]}
precision=$(python3 -c "from src.llm_speed_report import RUNGS; print(RUNGS['$deployed'][1])")
log "phase3: deployed rung $deployed"
if server_up "$dir" "$flags"; then
    python3 -u -m src.load_test --served-name $NAME --model-dir $MODEL --kind $KIND \
        --precision "$precision" --server-flags "$SERVE_ENV vllm serve $dir --served-model-name $NAME $flags" \
        --gpu-usd-per-hour $PRICE --price-source "$PRICE_SOURCE" > "$LOGS/load-test.log" 2>&1
    log "phase3: load test exit $?"
else
    log "phase3: load test failed, server did not come up"
fi
server_down

# 5. Phase 4 Task 11 Steps 2-3: the reference and the deployed rung on test, once.
speed hf_bf16_batched --split test --final && log "phase3: test reference exit 0" || log "phase3: test reference failed"
if server_up "$dir" "$flags"; then
    speed "$deployed" --served-name $NAME --split test --final
    log "phase3: test deployed exit $?"
else
    log "phase3: test deployed failed, server did not come up"
fi
server_down
log "PHASE3 DONE"

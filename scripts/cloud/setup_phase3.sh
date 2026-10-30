#!/usr/bin/env bash
# Plan 13 Phase 3 on a fresh rented GPU: unpack, install, build the serving and
# quantisation environments, then hand over to phase3.sh.
#
#   nohup bash setup_phase3.sh > logs/setup.out 2>&1 &
#
# Expects /root/plan13/{phase3-data.tar,point-4b.tar,phase3.sh}.
set -uo pipefail
WORK=/root/plan13; REPO=$WORK/repo; LOGS=$WORK/logs
mkdir -p "$REPO" "$LOGS"; cd "$WORK"
log() { echo "$*  ($(date))" >> "$LOGS/run.log"; }

tar xf phase3-data.tar -C "$REPO" && tar xf "$REPO/code.tar" -C "$REPO" && rm "$REPO/code.tar" \
    && mkdir -p "$REPO/models/llm-rank" && tar xf point-4b.tar -C "$REPO/models/llm-rank" \
    || { log "setup: failed to unpack"; exit 1; }

# Triton compiles a launcher with gcc and the Python headers at first use.
if ! command -v gcc >/dev/null 2>&1 || ! python3 -c "import sysconfig,os,sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc python3-dev >/dev/null \
        || { log "setup: failed, could not install gcc"; exit 1; }
fi

python3 -m pip install --quiet uv "transformers==5.17.0" pandas pyarrow numpy scipy zstandard \
    "sentence-transformers==6.1.0" "lightgbm==4.7.0" "datasets>=3.0" "accelerate>=1.1.0" 2>&1 | tail -2

cd "$REPO"
export PYTHONPATH="$REPO" UV_EXCLUDE_NEWER=2026-10-16T00:00:00Z
python3 -c "import torch, transformers, src.llm_speed_report, src.load_test, src.serve_client; print('project', torch.__version__, transformers.__version__, torch.cuda.get_device_name())" \
    >> "$LOGS/build-serve.log" 2>&1 || { log "setup: failed, project imports"; exit 1; }

{ uv venv --python 3.12 .venv-serve && uv pip install --python .venv-serve/bin/python vllm \
  && .venv-serve/bin/python -c "import vllm; print('vllm', vllm.__version__)"; } >> "$LOGS/build-serve.log" 2>&1 \
    || { log "setup: failed, vllm env"; exit 1; }
{ uv venv --python 3.12 .venv-quant && uv pip install --python .venv-quant/bin/python "llmcompressor==0.9.0.5" pandas pyarrow datasets \
  && .venv-quant/bin/python -c "import llmcompressor, compressed_tensors, torch, transformers; print('llmcompressor', llmcompressor.__version__, 'compressed-tensors', compressed_tensors.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" \
  && echo "QUANT ENV READY"; } > "$LOGS/build-quant.log" 2>&1 \
    || { log "setup: failed, quantisation env"; exit 1; }
log "setup: done"

cd "$WORK" && exec bash phase3.sh > "$LOGS/phase3.out" 2>&1

#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_path="$(find "$root_dir/.tools/llama.cpp" -type f -name llama-server -print -quit)"
cudart_path="$(find "$root_dir/.tools/llama.cpp" -type f -name libcudart.so.12 -print -quit)"
model_path="${MODEL_PATH:-$root_dir/models/granite-4.2-8b-Q4_K_M.gguf}"
port="${LLAMA_PORT:-8080}"
ctx_size="${LLAMA_CTX_SIZE:-8192}"
gpu_layers="${LLAMA_GPU_LAYERS:-999}"
reasoning="${LLAMA_ARG_REASONING:-auto}"
reasoning_format="${LLAMA_ARG_THINK:-auto}"

if [[ -z "$server_path" || ! -x "$server_path" ]]; then
  echo "llama-server is missing; run scripts/setup_runtime.sh" >&2
  exit 1
fi
if [[ ! -s "$model_path" ]]; then
  echo "model is missing: $model_path" >&2
  exit 1
fi

mkdir -p "$root_dir/logs"
if [[ -z "$cudart_path" ]]; then
  echo "CUDA runtime library is missing; run scripts/setup_runtime.sh" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$(dirname "$server_path"):$(dirname "$cudart_path"):${LD_LIBRARY_PATH:-}"

exec "$server_path" \
  --model "$model_path" \
  --host 127.0.0.1 \
  --port "$port" \
  --ctx-size "$ctx_size" \
  --n-gpu-layers "$gpu_layers" \
  --parallel 1 \
  --jinja \
  --reasoning "$reasoning" \
  --reasoning-format "$reasoning_format" \
  --metrics

#!/usr/bin/env bash
# =============================================================================
# docker-build.sh — Helper script for building and running Qwen Image API
# =============================================================================
set -euo pipefail

IMAGE="qwen-image-api:latest"
CONTAINER="qwen-image-api"
PORT=8190

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  build          Build the Docker image (multi-arch: 8.0;8.6;8.9;9.0;10.0;12.0)
  build-fast     Build for ONLY sm_120 — shorter compile time, RTX Pro 6000 only
  up             Start container with docker compose
  down           Stop and remove container
  logs           Tail container logs
  shell          Open a bash shell inside the running container
  health         Check the /health endpoint
  test-text2img  Send a test text-to-image request
  test-edit      Send a test edit request (requires test.png)
  clean          Remove image and container

Environment variables:
  HF_TOKEN       Hugging Face token (if model repo requires authentication)
  http_proxy     Corporate proxy URL (if required, e.g. http://proxy.intra:80)
  https_proxy    Corporate proxy URL (same as http_proxy usually)
EOF
}

PROXY_ARGS=(
  --build-arg http_proxy="http://proxy.intra:80"
  --build-arg https_proxy="http://proxy.intra:80"
  --build-arg no_proxy="localhost,127.0.0.1"
)

build() {
  echo "🔨 Building $IMAGE (all architectures: 8.0;8.6;8.9;9.0;10.0;12.0)..."
  docker build \
    "${PROXY_ARGS[@]}" \
    --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    --build-arg MAX_JOBS="$(nproc)" \
    -t "$IMAGE" .
  echo "✅ Build complete: $IMAGE"
}

build_fast() {
  echo "⚡ Building $IMAGE (sm_120 only — RTX Pro 6000 target)..."
  docker build \
    "${PROXY_ARGS[@]}" \
    --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
    --build-arg MAX_JOBS="$(nproc)" \
    -t "$IMAGE" .
  echo "✅ Fast build complete: $IMAGE"
}

up() {
  docker compose up -d
  echo "🚀 Container started."
  echo "   API docs : http://localhost:$PORT/docs"
  echo "   Health   : http://localhost:$PORT/health"
}

down() {
  docker compose down
}

logs() {
  docker compose logs -f "$CONTAINER"
}

shell() {
  docker exec -it "$CONTAINER" bash
}

health() {
  curl -s "http://localhost:$PORT/health" | python3 -m json.tool
}

test_text2img() {
  echo "📤 Sending test text-to-image request..."
  curl -s -X POST "http://localhost:$PORT/text2img" \
    -F "prompt=a cute cat sitting on a wooden table, photorealistic" \
    -F "aspect_ratio=1:1" \
    -F "num_steps=30" \
    -F "num_samples=1" \
    | python3 -m json.tool
}

test_edit() {
  if [[ ! -f test.png ]]; then
    echo "❌ test.png not found. Place an image named test.png in the project root."
    exit 1
  fi
  echo "📤 Sending test edit request with test.png..."
  curl -s -X POST "http://localhost:$PORT/edit" \
    -F "file=@test.png" \
    -F "prompt=make the background white" \
    -F "num_samples=1" \
    | python3 -m json.tool
}

clean() {
  read -p "⚠️  This removes the container and image. Continue? [y/N] " yn
  case $yn in
    [Yy]*)
      docker compose down 2>/dev/null || true
      docker rmi "$IMAGE" 2>/dev/null || true
      echo "🧹 Cleaned."
      ;;
    *) echo "Aborted." ;;
  esac
}

case "${1:-}" in
  build)         build ;;
  build-fast)    build_fast ;;
  up)            up ;;
  down)          down ;;
  logs)          logs ;;
  shell)         shell ;;
  health)        health ;;
  test-text2img) test_text2img ;;
  test-edit)     test_edit ;;
  clean)         clean ;;
  *)             usage ;;
esac

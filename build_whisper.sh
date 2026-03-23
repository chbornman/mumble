#!/usr/bin/env bash
# build_whisper.sh - Build whisper.cpp with settings from config.toml
#
# Usage:
#   ./build_whisper.sh                  # Build using config.toml settings
#   ./build_whisper.sh --backend cpu    # Override backend
#   ./build_whisper.sh --backend vulkan # Build with Vulkan GPU support
#   ./build_whisper.sh --clean          # Clean build directory first

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Defaults (will be overridden by config)
BACKEND_OVERRIDE=""
CLEAN=false

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)  BACKEND_OVERRIDE="$2"; shift 2 ;;
        --clean)    CLEAN=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend cpu|vulkan  Override backend from config.toml"
            echo "  --clean               Clean build directory before building"
            echo "  --help                Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Read config
eval "$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from config_loader import load_config
c = load_config('$SCRIPT_DIR/config.toml')
print(f'WHISPER_CPP_DIR=\"{c.paths.whisper_cpp_dir}\"')
print(f'BACKEND=\"{c.backend.type}\"')
print(f'BUILD_TYPE=\"{c.build.build_type}\"')
print(f'BUILD_SERVER={str(c.build.build_server).lower()}')
print(f'BUILD_SDL2={str(c.build.build_sdl2).lower()}')
print(f'CPU_NATIVE={str(c.backend.cpu.native).lower()}')
print(f'CPU_OPENMP={str(c.backend.cpu.openmp).lower()}')
print(f'CPU_LTO={str(c.backend.cpu.lto).lower()}')
print(f'CPU_REPACK={str(c.backend.cpu.repack).lower()}')
print(f'CPU_BLAS={str(c.backend.cpu.blas).lower()}')
print(f'VULKAN_DEVICE=\"{c.backend.vulkan.device}\"')
" 2>/dev/null)"

if [ -z "$WHISPER_CPP_DIR" ]; then
    log_error "Could not read config.toml"
    exit 1
fi

# Apply override
if [ -n "$BACKEND_OVERRIDE" ]; then
    BACKEND="$BACKEND_OVERRIDE"
fi

log_info "Building whisper.cpp"
log_info "  Directory: $WHISPER_CPP_DIR"
log_info "  Backend:   $BACKEND"
log_info "  Build:     $BUILD_TYPE"

# Verify directory exists
if [ ! -d "$WHISPER_CPP_DIR" ]; then
    log_error "whisper.cpp not found at $WHISPER_CPP_DIR"
    log_info "Clone it with: git clone https://github.com/ggerganov/whisper.cpp.git $WHISPER_CPP_DIR"
    exit 1
fi

BUILD_DIR="$WHISPER_CPP_DIR/build"

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Helper to convert bool to cmake ON/OFF
cmake_bool() {
    if [ "$1" = "true" ]; then echo "ON"; else echo "OFF"; fi
}

# Build cmake args
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DGGML_NATIVE="$(cmake_bool $CPU_NATIVE)"
    -DGGML_OPENMP="$(cmake_bool $CPU_OPENMP)"
    -DGGML_LTO="$(cmake_bool $CPU_LTO)"
    -DGGML_CPU_REPACK="$(cmake_bool $CPU_REPACK)"
    -DGGML_BLAS="$(cmake_bool $CPU_BLAS)"
    -DWHISPER_BUILD_SERVER="$(cmake_bool $BUILD_SERVER)"
    -DWHISPER_SDL2="$(cmake_bool $BUILD_SDL2)"
)

# Backend-specific flags
if [ "$BACKEND" = "vulkan" ]; then
    CMAKE_ARGS+=(
        -DGGML_VULKAN=ON
        -DGGML_CUDA=OFF
    )
    log_info "  Vulkan GPU acceleration enabled (device $VULKAN_DEVICE)"
elif [ "$BACKEND" = "cpu" ]; then
    CMAKE_ARGS+=(
        -DGGML_VULKAN=OFF
        -DGGML_CUDA=OFF
    )
    log_info "  CPU-only build"
else
    log_error "Unknown backend: $BACKEND (expected 'cpu' or 'vulkan')"
    exit 1
fi

# Ensure all other GPU backends are off (unless explicitly set)
CMAKE_ARGS+=(
    -DGGML_METAL=OFF
    -DGGML_HIP=OFF
    -DGGML_OPENCL=OFF
)

log_info "Running cmake configure..."
cmake -S "$WHISPER_CPP_DIR" "${CMAKE_ARGS[@]}"

log_info "Building (using $(nproc) threads)..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j"$(nproc)"

# Verify build
BINARIES=("whisper-cli" "whisper-stream")
if [ "$BUILD_SERVER" = "true" ]; then
    BINARIES+=("whisper-server")
fi

echo ""
log_ok "Build complete! Binaries:"
for bin in "${BINARIES[@]}"; do
    bin_path="$BUILD_DIR/bin/$bin"
    if [ -f "$bin_path" ]; then
        size=$(du -h "$bin_path" | cut -f1)
        log_ok "  $bin ($size)"
    else
        log_warn "  $bin - NOT FOUND"
    fi
done

# Show linked GPU libraries
echo ""
log_info "Linked libraries (GPU check):"
ldd "$BUILD_DIR/bin/whisper-cli" 2>/dev/null | grep -iE "vulkan|cuda|hip|opencl|ggml" || log_info "  (CPU only - no GPU libraries linked)"

echo ""
log_ok "Done. Backend: $BACKEND"

#!/usr/bin/env bash
# install.sh - Install whisper dictation daemon
# Reads settings from config.toml and sets up the full environment.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DAEMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$DAEMON_DIR/config.toml"
DRY_RUN=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)     DRY_RUN=true; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Show what would be done without making changes"
            echo "  --skip-build   Skip building whisper.cpp (use existing build)"
            echo "  --help         Show this help message"
            echo ""
            echo "Configuration is read from config.toml. Edit it before running."
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Logging
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would run: $*"
    else
        "$@"
    fi
}

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    log_info "Copy config.toml.example to config.toml and edit it first."
    exit 1
fi

# Read config values via Python
log_info "Reading configuration from config.toml..."
eval "$(python3 -c "
import sys
sys.path.insert(0, '$DAEMON_DIR')
from config_loader import load_config
c = load_config('$CONFIG_FILE')
print(f'WHISPER_CPP_DIR=\"{c.paths.whisper_cpp_dir}\"')
print(f'MODELS_DIR=\"{c.paths.models_dir}\"')
print(f'MODEL_NAME=\"{c.model.name}\"')
print(f'BACKEND=\"{c.backend.type}\"')
" 2>/dev/null)" || { log_error "Failed to parse config.toml"; exit 1; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    local missing=()
    local required_cmds=("wtype" "git" "cmake" "gcc" "uv" "python3" "ncat")

    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if ! command -v waybar &>/dev/null; then
        log_warn "waybar not found - status indicator will not work"
    fi

    if ! command -v pw-play &>/dev/null; then
        log_warn "pw-play not found - audio feedback will not work"
    fi

    if [ "$BACKEND" = "vulkan" ]; then
        if ! command -v vulkaninfo &>/dev/null; then
            log_warn "vulkaninfo not found - Vulkan backend may not work"
            log_info "Install vulkan-tools to verify GPU support"
        fi
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo "Install them with your package manager."
        return 1
    fi

    log_success "All required dependencies found"
}

# Setup whisper.cpp
setup_whisper_cpp() {
    log_info "Setting up whisper.cpp in $WHISPER_CPP_DIR..."

    if [ -d "$WHISPER_CPP_DIR" ]; then
        if [ -f "$WHISPER_CPP_DIR/build/bin/whisper-cli" ] && [ "$SKIP_BUILD" = true ]; then
            log_success "whisper.cpp already built (skipping)"
            return 0
        fi
    else
        log_info "Cloning whisper.cpp..."
        run_cmd mkdir -p "$(dirname "$WHISPER_CPP_DIR")"
        run_cmd git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_CPP_DIR"
    fi

    if [ "$SKIP_BUILD" = false ]; then
        log_info "Building whisper.cpp (backend: $BACKEND)..."
        if [ "$DRY_RUN" = false ]; then
            bash "$DAEMON_DIR/build_whisper.sh"
        else
            echo -e "${YELLOW}[DRY-RUN]${NC} Would run: bash $DAEMON_DIR/build_whisper.sh"
        fi
    fi

    log_success "whisper.cpp ready"
}

# Download model
download_model() {
    log_info "Checking model: $MODEL_NAME..."

    local model_file="$MODELS_DIR/ggml-$MODEL_NAME.bin"

    if [ -f "$model_file" ]; then
        log_success "Model already downloaded: $(basename $model_file)"
        return 0
    fi

    log_info "Downloading model: $MODEL_NAME..."
    if [ "$DRY_RUN" = false ]; then
        cd "$WHISPER_CPP_DIR"
        bash ./models/download-ggml-model.sh "$MODEL_NAME"
        cd "$DAEMON_DIR"
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} Would download model: $MODEL_NAME"
    fi

    log_success "Model downloaded"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."

    if [ ! -d "$DAEMON_DIR/.venv" ]; then
        run_cmd uv venv "$DAEMON_DIR/.venv"
    fi

    run_cmd uv pip install -r "$DAEMON_DIR/requirements.txt"
    log_success "Python environment ready"
}

# Install systemd service
install_service() {
    log_info "Installing systemd user service..."

    local service_dir="$HOME/.config/systemd/user"
    run_cmd mkdir -p "$service_dir"

    if [ "$DRY_RUN" = false ]; then
        cp "$DAEMON_DIR/whisper.service" "$service_dir/whisper.service"
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} Would copy whisper.service"
    fi

    run_cmd systemctl --user daemon-reload
    run_cmd systemctl --user enable whisper.service
    run_cmd systemctl --user start whisper.service

    log_success "Service installed and started"
}

# Make scripts executable
make_executable() {
    log_info "Setting script permissions..."
    chmod +x "$DAEMON_DIR"/*.sh "$DAEMON_DIR"/*.py 2>/dev/null || true
    log_success "Scripts are executable"
}

# Test installation
test_installation() {
    log_info "Testing installation..."

    if systemctl --user is-active --quiet whisper.service; then
        log_success "Whisper daemon is running"
    else
        log_error "Whisper daemon is not running"
        echo "Check logs: journalctl --user -u whisper.service -n 50"
        return 1
    fi

    log_success "Installation test passed"
}

# Main
main() {
    echo ""
    echo "=========================================="
    echo "  Whisper Dictation Daemon Installer"
    echo "=========================================="
    echo ""
    echo "  Backend: $BACKEND"
    echo "  Model:   $MODEL_NAME"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY RUN MODE - No changes will be made"
        echo ""
    fi

    check_prerequisites
    setup_whisper_cpp
    download_model
    setup_python_env
    make_executable
    install_service

    if [ "$DRY_RUN" = false ]; then
        test_installation
    fi

    echo ""
    echo "=========================================="
    log_success "Installation complete!"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  SUPER+D         - Toggle streaming mode"
    echo "  SUPER+Shift+D   - Toggle recording (dictation)"
    echo "  Right-click      waybar indicator - Switch model"
    echo ""
    echo "Config:   $CONFIG_FILE"
    echo "Logs:     journalctl --user -u whisper.service -f"
    echo "Benchmark: python3 benchmark.py"
    echo ""
}

main

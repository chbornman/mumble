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
WITH_STREAMING=false
DOWNLOAD_MODEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=true; shift ;;
        --skip-build)     SKIP_BUILD=true; shift ;;
        --with-streaming) WITH_STREAMING=true; shift ;;
        --download-model) DOWNLOAD_MODEL=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show what would be done without making changes"
            echo "  --skip-build      Skip building whisper.cpp (use existing build)"
            echo "  --with-streaming  Set up the OPT-IN Nemotron streaming sidecar"
            echo "                    (heavy NeMo venv probe + mumble-stt.service)."
            echo "                    Implied when config streaming_backend/type is"
            echo "                    \"nemotron-streaming\". Downloads ~several GB on"
            echo "                    first sidecar start."
            echo "  --download-model  With --with-streaming: warm the HF model cache"
            echo "                    during install instead of on first sidecar start."
            echo "  --help            Show this help message"
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
import sys, tomllib
sys.path.insert(0, '$DAEMON_DIR')
from config_loader import load_config
c = load_config('$CONFIG_FILE')
print(f'WHISPER_CPP_DIR=\"{c.paths.whisper_cpp_dir}\"')
print(f'MODELS_DIR=\"{c.paths.models_dir}\"')
print(f'MODEL_NAME=\"{c.model.name}\"')
print(f'BACKEND=\"{c.backend.type}\"')
print(f'STREAMING_BACKEND=\"{getattr(c.backend, \"streaming_backend\", \"\")}\"')

# The sidecar reads its own [mumble_stt] table directly (config_loader does not
# model it). Parse the raw TOML for the few knobs the installer needs.
with open('$CONFIG_FILE', 'rb') as f:
    raw = tomllib.load(f)
m = raw.get('mumble_stt', {}) if isinstance(raw.get('mumble_stt'), dict) else {}
default_venv = '$DAEMON_DIR/.venv-stt/bin/python'
print(f'STT_VENV_PYTHON=\"{m.get(\"venv_python\", default_venv)}\"')
print(f'STT_MODEL=\"{m.get(\"model\", \"nvidia/nemotron-speech-streaming-en-0.6b\")}\"')
" 2>/dev/null)" || { log_error "Failed to parse config.toml"; exit 1; }

# Opt into the heavy streaming setup either via --with-streaming or when the
# config already selects the nemotron-streaming backend. (backend.type is the
# batch selector; streaming_backend is the streaming selector — honor either
# landing on "nemotron-streaming" so config and flag stay consistent.)
if [ "$STREAMING_BACKEND" = "nemotron-streaming" ] || [ "$BACKEND" = "nemotron-streaming" ]; then
    WITH_STREAMING=true
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    local missing=()
    local required_cmds=("git" "cmake" "gcc" "uv" "python3" "ncat")

    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    # Text injection is a pluggable backend (config wayland.typer): any ONE of
    # wtype (Wayland), xdotool (X11), or ydotool satisfies the requirement.
    # Detect + report only; never auto-install with the user's package manager.
    local injector_found=""
    for inj in wtype xdotool ydotool; do
        if command -v "$inj" &>/dev/null; then
            injector_found="$inj"
            break
        fi
    done
    if [ -z "$injector_found" ]; then
        log_error "No text injector found (need one of: wtype, xdotool, ydotool)"
        echo "Install one for your session and package manager, e.g.:"
        echo "  Arch:    sudo pacman -S wtype     # Wayland (xdotool for X11)"
        echo "  Debian:  sudo apt install wtype   # Wayland (xdotool for X11)"
        echo "  Fedora:  sudo dnf install wtype   # Wayland (xdotool for X11)"
        echo "  X11 sessions: use xdotool instead of wtype."
        return 1
    fi
    log_success "Text injector found: $injector_found"

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
        cp "$DAEMON_DIR/mumble.service" "$service_dir/mumble.service"
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} Would copy mumble.service"
    fi

    run_cmd systemctl --user daemon-reload
    run_cmd systemctl --user enable mumble.service
    run_cmd systemctl --user start mumble.service

    log_success "Service installed and started"
}

# Setup the OPT-IN Nemotron streaming sidecar (mumble-stt).
#
# Heavy, GPU-only, multi-GB. Never runs unless opted in (--with-streaming or a
# config backend already set to "nemotron-streaming"). Per the "no magic" design
# goal: this probes and REPORTS; it does not build the heavy venv, does not
# download the model unless --download-model is passed, and does not enable or
# start the service (it prints the copy-paste command instead).
setup_mumble_stt() {
    log_info "Setting up Nemotron streaming sidecar (mumble-stt)..."

    # 1. Verify the heavy venv exists. We do NOT build it silently.
    if [ ! -x "$STT_VENV_PYTHON" ]; then
        log_error "Heavy ASR venv not found at: $STT_VENV_PYTHON"
        log_info "Build it (from the repo root):"
        log_info "    python -m venv .venv-stt"
        log_info "    .venv-stt/bin/pip install -r mumble_stt/requirements.txt"
        log_info "or point [mumble_stt].venv_python at an existing NeMo venv."
        return 1
    fi
    log_success "Heavy venv python found: $STT_VENV_PYTHON"

    # 2. Probe the venv non-destructively: report torch/NeMo versions + CUDA.
    log_info "Probing heavy venv (torch / NeMo / CUDA)..."
    if [ "$DRY_RUN" = false ]; then
        if "$STT_VENV_PYTHON" -c "import torch, nemo.collections.asr as a; import nemo; print(f'torch {torch.__version__}, NeMo {nemo.__version__}, cuda={torch.cuda.is_available()}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))" 2>/tmp/mumble_stt_probe.err; then
            :
        else
            log_warn "Heavy deps probe failed (torch/nemo import error):"
            sed 's/^/    /' /tmp/mumble_stt_probe.err 2>/dev/null | tail -n 5
            log_info "The sidecar will not start until the venv imports cleanly."
        fi
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} Would probe: $STT_VENV_PYTHON -c 'import torch, nemo.collections.asr; print(torch.cuda.is_available())'"
    fi

    # 3. Model: only warm the HF cache during install if explicitly requested.
    #    Otherwise the first sidecar start downloads it.
    if [ "$DOWNLOAD_MODEL" = true ]; then
        log_info "Pre-downloading model into the HF cache: $STT_MODEL..."
        if [ "$DRY_RUN" = false ]; then
            if "$STT_VENV_PYTHON" -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('$STT_MODEL')" 2>/tmp/mumble_stt_model.err; then
                log_success "Model cached: $STT_MODEL"
            else
                log_warn "Model pre-download failed (will retry on first sidecar start):"
                sed 's/^/    /' /tmp/mumble_stt_model.err 2>/dev/null | tail -n 5
            fi
        else
            echo -e "${YELLOW}[DRY-RUN]${NC} Would download model: $STT_MODEL"
        fi
    else
        log_info "Model $STT_MODEL will be downloaded on first sidecar start"
        log_info "(pass --download-model to warm the cache now)."
    fi

    # 4. Install the user unit (modeled on install_service / mumble.service, not
    #    the stale whisper.service name). Copy + daemon-reload only; do NOT
    #    enable/start a heavy GPU unit — print the command for the user instead.
    log_info "Installing mumble-stt.service user unit..."
    local service_dir="$HOME/.config/systemd/user"
    run_cmd mkdir -p "$service_dir"
    if [ "$DRY_RUN" = false ]; then
        cp "$DAEMON_DIR/mumble-stt.service" "$service_dir/mumble-stt.service"
    else
        echo -e "${YELLOW}[DRY-RUN]${NC} Would copy mumble-stt.service"
    fi
    run_cmd systemctl --user daemon-reload

    log_success "Sidecar unit installed (not enabled/started)."
    echo ""
    log_info "To enable the sidecar (loads the resident model, uses GPU VRAM):"
    echo "    systemctl --user enable --now mumble-stt.service"
    echo "    journalctl --user -u mumble-stt.service -n 50    # if it misbehaves"
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

    if systemctl --user is-active --quiet mumble.service; then
        log_success "Whisper daemon is running"
    else
        log_error "Whisper daemon is not running"
        echo "Check logs: journalctl --user -u mumble.service -n 50"
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
    if [ "$WITH_STREAMING" = true ]; then
        echo "  Streaming: nemotron-streaming sidecar (opt-in, heavy)"
    fi
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

    if [ "$WITH_STREAMING" = true ]; then
        setup_mumble_stt || log_warn "Streaming sidecar setup incomplete (see messages above)."
    else
        log_info "Nemotron streaming sidecar available; re-run with --with-streaming to enable (downloads ~several GB)."
    fi

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
    echo "Logs:     journalctl --user -u mumble.service -f"
    echo "Benchmark: python3 benchmark.py"
    echo ""
}

main

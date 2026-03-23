# Whisper Dictation Daemon

Voice dictation system using whisper.cpp with a unified configuration file (`config.toml`).

## Features

- 🎯 **Fully configurable**: All settings in `config.toml` (backend, model, threads, etc.)
- ⚡ **Three modes**: CLI (press-to-record), Server (persistent model), Stream (live VAD)
- 📊 **Waybar integration**: Visual mode and status indicators
- 🔧 **CPU or GPU backends**: Choose `cpu` or `vulkan` in config
- 📝 **Quantized model support**: Use `base.en-q5_1` for faster inference
- 🔄 **Live streaming deduplication**: Python-based smart deduplication
- 📈 **Benchmarking suite**: Compare configurations with `benchmark.py`
- 🚫 **100% offline**: No cloud dependencies

## Quick Start

1. **Edit config.toml** (see [Configuration](#configuration))
2. **Build whisper.cpp**:  
   ```bash
   ./build_whisper.sh          # Uses config.toml settings
   ./build_whisper.sh --backend vulkan  # Override backend
   ```
3. **Install**:  
   ```bash
   ./install.sh
   ```
4. **Use**:  
   - `SUPER+D` - Toggle streaming mode  
   - `SUPER+Shift+D` - Toggle recording (dictation)  
   - Right-click waybar → Switch model/mode

## Configuration

All settings are in `config.toml`. Key sections:

### [model]
```toml
name = "base.en"           # Model name (ggml-base.en.bin)
language = "en"            # "en" for English-only, "auto" for multilingual
```

### [backend]
```toml
type = "cpu"               # "cpu" or "vulkan"
threads = 0                # 0 = auto (capped at max_threads)
max_threads = 12           # Max threads when auto-detecting

[backend.cpu]
native = true              # -march=native
openmp = true              # OpenMP threading
lto = true                 # Link-time optimization (~5-10% faster)
repack = true              # Weight repacking for cache
blas = false               # Enable OpenBLAS (benchmark to verify)

[backend.vulkan]
device = 0                 # GPU device index
cpu_fallback = true        # Fallback to CPU if Vulkan fails
```

### [daemon]
```toml
mode = "cli"               # "cli" or "server"
notifications = false      # Disable when using waybar
```

### [transcription]
```toml
strip_leading_artifacts = true
strip_patterns = ["^--\\s*", "^-\\s+"]  # Remove leading "--" or "- "
```

### [streaming]
```toml
step = 0                   # 0 = wait for speech via VAD
buffer_length = 30000      # Audio buffer length (ms)
keep = 200                 # Buffer overlap (ms)
vad_threshold = 0.6        # Voice activity threshold (0.0-1.0)
threads = 8                # Streaming threads
```

## Building whisper.cpp

The `build_whisper.sh` script reads `config.toml` and builds with the specified backend:

```bash
# Build with config.toml settings (default: CPU)
./build_whisper.sh

# Force Vulkan backend
./build_whisper.sh --backend vulkan

# Clean build first
./build_whisper.sh --clean
```

To enable Vulkan GPU acceleration:
1. Set `backend.type = "vulkan"` in config.toml
2. Ensure you have a Vulkan-capable GPU and drivers
3. Run `./build_whisper.sh --clean`

## Benchmarking

Record yourself reading the benchmark passages and compare configurations:

```bash
# 1. Record audio (save as WAV in benchmarks/audio/)
#    Use: pw-record benchmarks/audio/passage_1_technical.wav

# 2. Run benchmark
.venv/bin/python benchmark.py          # Full benchmark
.venv/bin/python benchmark.py --quick  # Quick test
.venv/bin/python benchmark.py --json   # JSON output
```

See [benchmark.py](benchmark.py) for full options.

## Modes

| Mode | Activation | Best For |
|------|------------|----------|
| **Stream** (▶) | `SUPER+D` (toggle) | Live transcription, hands-free |
| **CLI** (●) | `SUPER+Shift+D` (hold) | Quick dictation, lower memory |
| **Server** (◆) | Waybar → Menu | Frequent use, larger models |

## Model Recommendations

| Model | Size | Speed (Ryzen 9 9900X) | Use Case |
|-------|------|----------------------|----------|
| `tiny.en` | 75 MB | ~30x realtime | Commands, quick notes |
| `base.en` | 148 MB | ~11.5x realtime | **Default** - balanced |
| `base.en-q5_1` | ~75 MB | ~2x faster than base.en | Faster, minimal accuracy loss |
| `small.en` | 488 MB | ~4x realtime | Better accuracy |
| `medium.en` | 1.5 GB | ~2x realtime | Professional work |
| `large-v3-turbo` | 1.6 GB | ~1.5x realtime | Maximum accuracy |

## Waybar Integration

The waybar module shows:
- **Icon**: Current mode (● cli, ◆ server, ~ streaming)
- **Tooltip**: Model, backend, mode, controls
- **Click**: Left = toggle stream, Right = model/mode menu

## Troubleshooting

- **Leading `--` in transcriptions**: Fixed via `strip_leading_artifacts` in config.toml
- **Text not appearing**: Check `which wtype` and cursor is in a text field
- **No sound**: Verify `sounds/` directory contains `snare.wav` and `hihat.wav`
- **Streaming duplicates**: The Python deduplicator in `stream_dedup.py` handles this
- **Logs**: `journalctl --user -u whisper.service -f`

## Offline Use

Everything runs locally:
- Audio capture via PipeWire/ALSA
- Inference via whisper.cpp (CPU or Vulkan GPU)
- Text injection via wtype (Wayland)
- No internet required

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Core inference engine
- [Asahi Linux](https://asahilinux.org/) - Platform inspiration
- Built with inspiration from desktop whisper-dictation-daemon

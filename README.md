# mumble

whisper.cpp dictation daemon with per-machine config, Vulkan/CPU backends, and Waybar integration.

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

## Machine-Specific Tuning

The optimal config depends heavily on hardware. The same model/backend combo can perform very differently across machines due to GPU driver maturity, CPU architecture, and memory bandwidth.

### margo (Desktop - Ryzen 9 9900X / Radeon Pro V620)

| Setting | Value | Why |
|---------|-------|-----|
| Backend | `vulkan` | Mature AMD RDNA Vulkan driver, ggml shaders optimized for this arch |
| Model | `large-v3-turbo` | V620 handles it easily (~0.8s for any length) |
| Mode | `cli` | Fast enough that model reload is negligible |
| Vulkan device | `1` | Second GPU |

### asahi (MacBook Pro M1 Pro - Asahi Linux)

| Setting | Value | Why |
|---------|-------|-----|
| Backend | `cpu` | Apple GPU Vulkan driver (Asahi/Mesa) is immature for compute shaders. CPU with ARM NEON is faster. |
| Model | `base.en` | Best speed/accuracy tradeoff (~0.6s). `small.en` is ~1.7s, `large-v3-turbo` is ~7s on CPU. |
| Mode | `cli` | Low memory footprint, base.en loads fast |
| Threads | `10` | M1 Pro has 8P+2E cores |

**Why not Vulkan on Apple Silicon?** The M1 Pro GPU is powerful, but Vulkan on Asahi is a reverse-engineered driver translating a foreign API to Apple's unique GPU architecture. Compute shader performance (matrix multiplications for inference) lags behind the mature CPU NEON path. This may improve as the Asahi Mesa driver matures.

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

## Future Investigation

- **[Moonshine](https://github.com/usefulsensors/moonshine)** — MIT-licensed on-device speech-to-text model from Useful Sensors. Unlike Whisper (encoder-decoder, processes fixed 30s chunks), Moonshine is optimized for low-latency partial inference on chunks as short as ~1s. Could enable true real-time streaming without the buffer-fill-then-dump behavior of whisper-stream. Available as ONNX/PyTorch (no whisper.cpp-style C++ runtime yet). Sizes: Tiny (~190MB), Base (~400MB).

## whisper.cpp Patches

Streaming mode requires a patched `whisper-stream` binary. The upstream binary lacks `--device` for GPU selection, which is needed on multi-GPU systems (e.g. margo, where device 0 is the iGPU and device 1 is the V620).

Both machines track a `mumble-patches` branch in `~/projects/whisper.cpp` with this patch. When updating whisper.cpp:

```bash
cd ~/projects/whisper.cpp
git fetch origin
git rebase origin/master   # or whatever upstream branch
# resolve conflicts if any, then rebuild
cd build && cmake --build . --target whisper-stream -j$(nproc)
```

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Core inference engine
- [Asahi Linux](https://asahilinux.org/) - Platform inspiration
- Built with inspiration from desktop whisper-dictation-daemon

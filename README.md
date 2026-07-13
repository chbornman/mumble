# mumble

**Offline, local voice dictation for Linux.** Press a key, talk, and the text lands
wherever your cursor is. No cloud, no account, no telemetry — everything runs on
your machine.

mumble is a small daemon (a systemd **user** service) that ties together a few
swappable pieces behind two keybinds: **press-to-talk dictation** and **toggle
live streaming**. It is Linux-first and Wayland-first (developed on Hyprland;
Sway and others work the same way), with X11 supported by choosing an X11 text
injector.

Under the hood mumble is modular. The speech-to-text engine, the way text gets
typed into your apps, and an optional LLM cleanup pass are all **pluggable
backends** selected in `config.toml`. Out of the box it uses
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) (CPU or Vulkan GPU). If
you opt in, it can instead drive a real-time streaming engine built on NVIDIA's
**Nemotron 3.5** ASR for sub-second, natively-punctuated live transcription.

---

## Features

- **100% offline.** Audio never leaves the machine. No API keys, no network
  required after install.
- **Two actions, your keybinds.** Press-to-talk dictation and a live-streaming
  toggle. mumble never grabs keys itself — you bind them in your compositor (see
  [Usage](#usage--keybindings)).
- **Pluggable ASR backends.** whisper.cpp for batch press-to-talk (`whisper-cli`
  or a persistent `whisper-server`), plus a streaming path: the legacy
  `whisper-stream` pipeline, or an opt-in **Nemotron 3.5** streaming engine.
- **Pluggable text injection.** Works with `wtype` (Wayland), `xdotool` (X11), or
  `ydotool` (both) — mumble just needs one of them.
- **CPU or GPU.** Pick `cpu` or `vulkan` per machine; quantized models (e.g.
  `base.en-q5_1`) supported for speed.
- **Config-driven and machine-portable.** All behavior lives in `config.toml`,
  with per-machine overrides in `config.local.toml`. Clone, install, and it
  adapts to the hardware.
- **Optional LLM cleanup.** An OpenAI-compatible cleanup pass (strip fillers,
  fix punctuation, resolve self-corrections) exists but is **off by default**.
- **Waybar integration** for at-a-glance status, and audio feedback cues.
- **Honest installer.** Interactive, idempotent, and conservative — it never runs
  your package manager or `sudo` silently (see [Install](#install)).

---

## Requirements / Dependencies

mumble's Python dependencies are installed for you into a virtualenv by the
installer. A handful of dependencies, however, are **native programs that come
from your distro, not from pip** — a text injector, clipboard tools, `ffmpeg`,
and so on. The project's stance is deliberate: we **document and check** for
those, and print the exact install command for your distro, but we **never
silently install system packages or invoke `sudo`** on your behalf. You stay in
control of your own system.

### System packages (install these yourself)

You need:

- **A text injector — exactly one of:** `wtype` (Wayland, the default),
  `xdotool` (X11), or `ydotool` (works on both). Tell mumble which one via
  `wayland.typer` in `config.toml`.
- **`wl-clipboard`** (`wl-copy` / `wl-paste`) — used for the long-text paste path
  and for voice-command-on-selection. (On X11, the equivalent clipboard tooling.)
- **`ffmpeg`** — audio handling.
- **`ncat`** (from `openbsd-netcat`/`nmap`) — the keybind scripts talk to the
  daemon over a unix socket with `ncat -U`.
- **`git`, `cmake`, a C/C++ compiler** — to build whisper.cpp.
- **`uv`** — manages the Python virtualenv and dependencies.
- **A Vulkan driver** (`vulkan-tools` to verify) — only if you want GPU whisper
  (`backend.type = "vulkan"`). CPU works with no GPU at all.
- **Optional but recommended:** `waybar` (status module) and PipeWire's `pw-play`
  (audio feedback cues).

Copy-paste, per distro:

**Arch / pacman**
```bash
sudo pacman -S wtype wl-clipboard ffmpeg openbsd-netcat git cmake gcc uv \
               vulkan-tools waybar
# X11 instead of wtype:  sudo pacman -S xdotool   (or: ydotool)
```

**Debian / Ubuntu / apt**
```bash
sudo apt install wtype wl-clipboard ffmpeg ncat git cmake g++ \
                 vulkan-tools waybar
# uv:  see https://docs.astral.sh/uv/  (curl installer or pipx install uv)
# X11 instead of wtype:  sudo apt install xdotool   (or: ydotool)
```

**Fedora / dnf**
```bash
sudo dnf install wtype wl-clipboard ffmpeg nmap-ncat git cmake gcc-c++ \
                 vulkan-tools waybar uv
# X11 instead of wtype:  sudo dnf install xdotool   (or: ydotool)
```

> The installer detects what's missing and prints the right command for *your*
> distro, then stops. Nothing above is installed behind your back.

### Optional: Nemotron streaming engine (opt-in, heavy)

The real-time Nemotron 3.5 streaming backend runs as a separate **sidecar**
process on **NeMo + PyTorch**, which is multiple GB and pins its own CUDA torch —
so it can't share mumble's lightweight venv. It is strictly opt-in and only
pulled in if you ask for it (`install.sh --with-streaming`, or by selecting
`backend.streaming_backend = "nemotron-streaming"`). Requirements:

- An **NVIDIA GPU** with **CUDA** (driver new enough for the CUDA 12.8 wheels),
  or a fast CPU with enough DRAM. Select it with `[mumble_stt].device = "cuda"`
  or `"cpu"`; CPU inference uses float32 and is substantially slower but avoids
  keeping the model in VRAM.
- A **dedicated heavy venv** holding NeMo + torch (multi-GB). The installer
  *probes and reports* this — it does not build the heavy venv for you. Build
  it once, from the repo root:

  ```bash
  python -m venv .venv-stt            # or: uv venv .venv-stt
  .venv-stt/bin/pip install -r mumble_stt/requirements.txt
  ```

  If your heavy venv lives elsewhere, set `[mumble_stt].venv_python` in
  `config.local.toml` and update the `ExecStart` in `mumble-stt.service`.
- The model `nvidia/nemotron-speech-streaming-en-0.6b`, downloaded on first
  sidecar start (or warmed early with `--download-model`).

CPU mode is a practical memory-placement choice, not just an emergency
fallback. In an earlier direct-engine microbenchmark with the model already
loaded, an 11.0-second sample took 7.087 seconds on CPU versus 0.652 seconds on
CUDA (about 10.9x slower, but still 1.55x real time). Its paced direct-engine
run finished 156 ms after the audio ended. The CPU-resident sidecar used about
5.6 GB of DRAM and no idle VRAM. These measurements exclude some end-to-end
sidecar overhead measured below and are one-host observations, not portable
guarantees; benchmark your own workload before choosing a device.

See the sidecar sections of [ARCHITECTURE.md](ARCHITECTURE.md) for the design
and the wire protocol.

### Python dependencies

Handled automatically. `install.sh` creates a `.venv` and installs everything in
`requirements.txt` via `uv`. You don't pip-install anything by hand.

---

## Install

```bash
git clone https://github.com/chbornman/mumble.git
cd mumble
# 1. Edit config.toml for your machine (model, cpu/vulkan, injector). See below.
# 2. Run the installer.
./install.sh
```

`install.sh` is **interactive, idempotent, and conservative**. It:

- checks for required tools and **prints the exact install command** for anything
  missing on your distro, then stops — it does **not** run your package manager;
- creates the `.venv` and installs Python deps with `uv`;
- clones and builds **whisper.cpp** for your configured backend;
- **auto-downloads** the configured whisper model;
- installs and starts the **systemd user service**;
- prints the keybind commands to wire up in your compositor.

Useful flags:

```bash
./install.sh --dry-run          # show what it would do, change nothing
./install.sh --skip-build       # reuse an existing whisper.cpp build
./install.sh --with-streaming   # set up the opt-in Nemotron sidecar (downloads GBs)
./install.sh --download-model   # with --with-streaming: warm the model cache now
```

Manage the service like any systemd user unit:

```bash
systemctl --user status mumble.service
journalctl --user -u mumble.service -f
```

And when something doesn't work, run the one-command preflight:

```bash
./mumble doctor    # checks injector, clipboard, binaries, model, services, sockets
```

---

## Usage / Keybindings

mumble exposes two actions. It deliberately **does not bind keys for you** —
every compositor configures hotkeys differently, and reaching into your config
would be exactly the kind of "magic" this project avoids. You bind the keys; the
keybind runs the `mumble` CLI (which sends a verb to the daemon over its unix
socket):

- **Press-to-talk dictation** → `mumble toggle`
- **Toggle live streaming** → `mumble stream`

(`toggle_dictation.sh` and `toggle_stream.sh` remain as thin wrappers over the
same commands if you prefer the script paths.)

Pick whatever keys you like. The snippets below use `SUPER+Shift+D` for dictation
and `SUPER+D` for streaming, pointing at a clone in `~/projects/mumble` — adjust
the path to wherever you cloned the repo.

**Hyprland** (`~/.config/hypr/hyprland.conf`)
```ini
bind = SUPER SHIFT, D, exec, ~/projects/mumble/mumble toggle
bind = SUPER,       D, exec, ~/projects/mumble/mumble stream
```

**Sway** (`~/.config/sway/config`)
```
bindsym $mod+Shift+d exec ~/projects/mumble/mumble toggle
bindsym $mod+d       exec ~/projects/mumble/mumble stream
```

**i3** (`~/.config/i3/config`)
```
bindsym $mod+Shift+d exec --no-startup-id ~/projects/mumble/mumble toggle
bindsym $mod+d       exec --no-startup-id ~/projects/mumble/mumble stream
```

**GNOME** (Settings → Keyboard → Custom Shortcuts; or via `gsettings`)
Add two custom shortcuts whose commands are the full path to the `mumble`
script plus `toggle` / `stream`, and assign keys in the dialog.

**KDE Plasma** (System Settings → Shortcuts → Custom Shortcuts)
Add a *Command/URL* shortcut for each action (`/path/to/mumble toggle` and
`/path/to/mumble stream`), then assign a trigger key.

> Tip: symlink the CLI onto your PATH (`ln -s ~/projects/mumble/mumble
> ~/.local/bin/mumble`) and the binds become just `mumble toggle` /
> `mumble stream`.

---

## Backends & Configuration

`config.toml` is the control plane — every component reads it, nothing hardcodes
paths. Per-machine overrides go in `config.local.toml` (gitignored), which is
deep-merged on top (GPU index, model choice, endpoints, etc.). The modular pieces:

**ASR backend (batch, for press-to-talk)** — `[backend].type`
- `cpu` — whisper.cpp on CPU; always works, no GPU needed.
- `vulkan` — whisper.cpp on a Vulkan GPU; pick the device with
  `[backend.vulkan].device`.

Daemon mode (`[daemon].mode`) chooses *how* the batch model runs: `cli` loads the
model per request (low memory), `server` keeps it resident via `whisper-server`
(faster, more memory).

**Streaming backend (for the live toggle)** — `[backend].streaming_backend`
- `whisper-stream` — the legacy whisper.cpp rolling-buffer pipeline plus
  `stream_dedup.py` heuristics. Works today on CPU or Vulkan.
- `nemotron-streaming` — the opt-in Nemotron 3.5 sidecar: true cache-aware
  streaming with native punctuation, emitting clean FINAL segments over a unix
  socket. The daemon's backend is a thin client to it. Tuned under `[nemotron]`
  (including `inject_mode = "finals" | "live"` — phrase-at-a-time vs
  type-as-you-speak) and `[mumble_stt]`. See
  [ARCHITECTURE.md](ARCHITECTURE.md).

**Text injector** — `[wayland].typer`
- `wtype` (Wayland, default), `xdotool` (X11), or `ydotool` (both). mumble
  auto-switches to a `wl-copy` + Ctrl+V clipboard paste for long text (past
  `clipboard_paste_threshold`), where `wtype` gets flaky.

**Optional LLM cleanup** — `[llm_postprocess]`
- An OpenAI-compatible (`/v1/chat/completions`) cleanup pass for the
  press-to-talk path. **Off by default** (`enabled = false`). Any HTTP failure
  falls back to the raw transcript. Point `endpoint` at any local llama.cpp /
  Ollama / LM Studio / vLLM server.

### Vocabulary behavior

`[paths].vocab_file` points to the broad `vocab.txt` source used by batch
dictation. Its consumers use it differently:

- Batch whisper.cpp gets a case-insensitively deduplicated initial prompt,
  bounded to 896 characters. Later entries win when the list is too long, so
  keep current project and personal terms near the end. `benchmark.py` builds
  the prompt through the same path as the daemon.
- The optional LLM cleanup path parses literals, correction mappings, and
  quoted rules into a structured glossary when that feature is enabled.
- Nemotron can opt into a decoder-level RNNT boosting tree with
  `[mumble_stt].vocab_biasing_enabled = true`. Its separate default
  `vocab.streaming.txt` is a curated 77-phrase priority list, avoiding the
  overhead and false-positive surface of feeding the 407-phrase batch glossary
  into every streaming frame. The committed feature default remains `false`;
  margo enables the compact list in `config.local.toml`.

Nemotron biasing is static and global for the loaded sidecar: the phrase tree is
built at startup, applies to every streaming frame, and survives utterance
resets. Restart `mumble-stt.service` after changing the vocabulary or its bias
settings. On CPU it uses the PyTorch implementation with Triton and CUDA graph
decoding disabled.

The original live CPU smoke used a 371-phrase snapshot; the broad file now
resolves to 407 phrases. That load test is not an accuracy result. On
the same 11-second sample, biasing produced the identical
transcript both ways. Bias off took 11.322 seconds end to end (0.972x real-time
throughput) with a 1.024-second paced post-audio tail; bias on took 11.923
seconds (0.923x) with a 1.476-second tail. Keep it disabled until representative
user speech demonstrates better technical-term recall without unacceptable
over-biasing. The compact list reduces the scope substantially, but its term
recall still needs representative user speech before tuning the bias strength.

### Streaming silence gate

`[nemotron].vad_enabled = true` runs the bundled Silero VAD v5 model in the
light daemon on CPU. It follows the PhotoProof capture design: retain one second
of pre-roll, ship speech plus a 15-window hysteresis hang, then ship three more
seconds of silence so Nemotron can own endpointing and emit a FINAL. Only long
idle silence is withheld, which lets the resident CPU model stop consuming
roughly four cores between utterances. Initialization and inference errors fail
open, so VAD failure sends all audio instead of losing dictation.

Silero requires 512-sample windows at 16 kHz and a subtle 64-sample context
prefix (`[1, 576]` ONNX input); omitting that prefix silently produces near-zero
speech probabilities. The implementation preserves that context. A local
11.625-second speech-plus-silence smoke detected one start and one end, shipped
10.3 seconds, and spent 17.3 ms total in VAD inference (0.15% of audio duration).
Longer quiet periods are almost entirely withheld after the trailing window.

A few common settings:

```toml
[model]
name = "large-v3-turbo"   # or base.en, small.en, tiny.en, base.en-q5_1, …
language = "en"

[backend]
type = "vulkan"                       # cpu | vulkan  (batch press-to-talk)
streaming_backend = "whisper-stream"  # whisper-stream | nemotron-streaming

[daemon]
mode = "cli"                          # cli | server

[wayland]
typer = "wtype"                       # wtype | xdotool | ydotool
```

To switch a backend, change the value and restart the service
(`systemctl --user restart mumble.service`). `config.toml` is fully commented;
see [ARCHITECTURE.md](ARCHITECTURE.md) for the seams and the full design.

With `[nemotron].inject_mode = "live"`, seeing the last few words disappear and
be retyped is expected. PARTIAL hypotheses are replaceable, so the model may
revise them as more audio arrives; mumble reconciles the visible text to the
new hypothesis and then to the FINAL. This correction behavior is part of live
ASR and happens even when `[llm_postprocess].enabled = false` — it is not an LLM
rewriting the transcript.

Wispr-style cleanup is a separate optional stage. Vocabulary prompting and
RNNT biasing affect recognition; `[llm_postprocess]` edits completed batch
transcripts through an OpenAI-compatible endpoint. It remains off by default
and is not part of Nemotron's low-latency streaming path.

### Model recommendations

| Model | Size | Use case |
|-------|------|----------|
| `tiny.en` | 75 MB | Commands, quick notes |
| `base.en` | 148 MB | Balanced default (great on CPU) |
| `base.en-q5_1` | ~75 MB | Faster than base.en, minimal accuracy loss |
| `small.en` | 488 MB | Better accuracy |
| `medium.en` | 1.5 GB | Professional work |
| `large-v3-turbo` | 1.6 GB | Maximum accuracy (best on a capable GPU) |

The optimal model + backend is hardware-dependent. A fast GPU loves
`large-v3-turbo` on Vulkan; a CPU-only or immature-GPU-driver machine is usually
happier on `base.en` with `cpu`. Put those choices in `config.local.toml`.

---

## How it works

A long-lived daemon (`whisper_daemon.py`, run as `mumble.service`) owns a unix
socket, recording and streaming lifecycle, backend selection, the
transcribe → optional-cleanup → inject pipeline, and feedback (sounds, notify,
Waybar). Keybinds are tiny IPC clients that send it text verbs. ASR backends, the
text injector, and the optional LLM are swappable behind small documented
interfaces.

For the full picture — the component diagram, the IPC wire format, the sidecar
design and wire protocol, and exactly where the extension seams are — read
**[ARCHITECTURE.md](ARCHITECTURE.md)**. Open roadmap items live in
[TODO.md](TODO.md).

---

## Contributing

Contributions are welcome — adding an ASR backend, a text injector, or a
keybind client all happen through the documented seams without daemon surgery.
See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to set up the venv, run the
tests, and the project's no-magic-install philosophy.

---

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — core batch inference
- [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
  — optional streaming engine

---

## License

[MIT](LICENSE) © 2026 Caleb Bornman.

"""
Config loader for Whisper Daemon.
Reads config.toml and provides typed access to all settings.
"""

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _expand_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Expand ~ and env vars in path, optionally resolve relative to base_dir."""
    expanded = os.path.expanduser(os.path.expandvars(path_str))
    p = Path(expanded)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


@dataclass
class PathsConfig:
    whisper_cpp_dir: Path
    models_dir: Path
    build_dir: Path
    project_dir: Path
    vocab_file: Optional[Path]
    sound_dir: Path


@dataclass
class ModelConfig:
    name: str
    language: str

    @property
    def filename(self) -> str:
        return f"ggml-{self.name}.bin"


@dataclass
class CpuConfig:
    native: bool
    openmp: bool
    lto: bool
    repack: bool
    blas: bool


@dataclass
class VulkanConfig:
    device: int
    cpu_fallback: bool


@dataclass
class BackendConfig:
    type: str  # "cpu" or "vulkan"
    threads: int
    max_threads: int
    processors: int
    flash_attention: bool
    cpu: CpuConfig
    vulkan: VulkanConfig

    @property
    def effective_threads(self) -> int:
        if self.threads == 0:
            return min(os.cpu_count() or 4, self.max_threads)
        return self.threads


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int
    sound_sample_rate: int


@dataclass
class DaemonConfig:
    mode: str  # "cli" or "server"
    socket_path: str
    recording_flag: str
    streaming_flag: str
    stream_pid_file: str
    log_file: str
    notifications: bool
    server_port: int
    server_startup_timeout: int
    server_health_check_interval: float


@dataclass
class TranscriptionConfig:
    temperature: float
    temperature_increment: float
    cli_timeout: int
    server_timeout: int
    strip_leading_artifacts: bool
    strip_patterns: list[str]
    response_format: str

    def clean_text(self, text: str) -> str:
        """Strip leading artifacts from transcribed text."""
        if not self.strip_leading_artifacts or not text:
            return text
        cleaned = text
        for pattern in self.strip_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip()


@dataclass
class StreamingDebugConfig:
    log_file: str
    output_log: str
    stream_log: str


@dataclass
class StreamingConfig:
    step: int
    buffer_length: int
    keep: int
    vad_threshold: float
    threads: int
    max_committed_length: int
    min_overlap_chars: int
    overlap_step: int
    fallback_suffix_length: int
    drift_reset_threshold: int
    max_fallback_sentence_length: int
    noise_filter_pattern: str
    debug: StreamingDebugConfig


@dataclass
class WaylandConfig:
    display: str
    xdg_runtime_dir: str
    typer: str
    notifier: str
    notification_timeout: int


@dataclass
class WaybarIconsConfig:
    cli_ready: str
    cli_recording: str
    cli_processing: str
    server_ready: str
    server_recording: str
    server_processing: str
    streaming: str
    error: str


@dataclass
class WaybarConfig:
    update_interval: float
    icons: WaybarIconsConfig


@dataclass
class BuildConfig:
    build_type: str
    build_server: bool
    build_sdl2: bool


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig
    backend: BackendConfig
    audio: AudioConfig
    daemon: DaemonConfig
    transcription: TranscriptionConfig
    streaming: StreamingConfig
    wayland: WaylandConfig
    waybar: WaybarConfig
    build: BuildConfig

    @property
    def model_path(self) -> Path:
        return self.paths.models_dir / self.model.filename

    @property
    def whisper_cli_path(self) -> Path:
        return self.paths.build_dir / "whisper-cli"

    @property
    def whisper_server_path(self) -> Path:
        return self.paths.build_dir / "whisper-server"

    @property
    def whisper_stream_path(self) -> Path:
        return self.paths.build_dir / "whisper-stream"

    @property
    def whisper_bench_path(self) -> Path:
        return self.paths.build_dir / "whisper-bench"


def _find_config_file(explicit_path: Optional[str] = None) -> Path:
    """Find config.toml in standard locations."""
    if explicit_path:
        p = Path(os.path.expanduser(explicit_path))
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    # Search order
    candidates = [
        Path.cwd() / "config.toml",
        Path(__file__).parent / "config.toml",
        Path.home() / ".config" / "whisper-daemon" / "config.toml",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"config.toml not found in any of: {[str(c) for c in candidates]}"
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """Load and parse config.toml into a Config dataclass."""
    path = _find_config_file(config_path).resolve()
    project_dir = path.parent

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Paths
    whisper_cpp_dir = _expand_path(raw["paths"]["whisper_cpp_dir"])
    models_dir = _expand_path(raw["paths"]["models_dir"], whisper_cpp_dir)
    build_dir = _expand_path(raw["paths"]["build_dir"], whisper_cpp_dir)
    explicit_project_dir = raw["paths"].get("project_dir")
    if explicit_project_dir:
        project_dir = _expand_path(explicit_project_dir)

    vocab_file_str = raw["paths"].get("vocab_file")
    vocab_file = _expand_path(vocab_file_str, project_dir) if vocab_file_str else None

    sound_dir = _expand_path(raw["paths"]["sound_dir"], project_dir)

    paths = PathsConfig(
        whisper_cpp_dir=whisper_cpp_dir,
        models_dir=models_dir,
        build_dir=build_dir,
        project_dir=project_dir,
        vocab_file=vocab_file,
        sound_dir=sound_dir,
    )

    # Model
    model_raw = raw["model"]
    model = ModelConfig(
        name=model_raw["name"],
        language=model_raw["language"],
    )

    # Backend
    backend_raw = raw["backend"]
    cpu_raw = backend_raw.get("cpu", {})
    vulkan_raw = backend_raw.get("vulkan", {})

    cpu = CpuConfig(
        native=cpu_raw.get("native", True),
        openmp=cpu_raw.get("openmp", True),
        lto=cpu_raw.get("lto", True),
        repack=cpu_raw.get("repack", True),
        blas=cpu_raw.get("blas", False),
    )

    vulkan = VulkanConfig(
        device=vulkan_raw.get("device", 0),
        cpu_fallback=vulkan_raw.get("cpu_fallback", True),
    )

    backend = BackendConfig(
        type=backend_raw["type"],
        threads=backend_raw.get("threads", 0),
        max_threads=backend_raw.get("max_threads", 12),
        processors=backend_raw.get("processors", 1),
        flash_attention=backend_raw.get("flash_attention", False),
        cpu=cpu,
        vulkan=vulkan,
    )

    # Audio
    audio_raw = raw["audio"]
    audio = AudioConfig(
        sample_rate=audio_raw["sample_rate"],
        channels=audio_raw["channels"],
        sound_sample_rate=audio_raw["sound_sample_rate"],
    )

    # Daemon
    daemon_raw = raw["daemon"]
    daemon = DaemonConfig(
        mode=daemon_raw["mode"],
        socket_path=daemon_raw["socket_path"],
        recording_flag=daemon_raw["recording_flag"],
        streaming_flag=daemon_raw["streaming_flag"],
        stream_pid_file=daemon_raw["stream_pid_file"],
        log_file=daemon_raw["log_file"],
        notifications=daemon_raw["notifications"],
        server_port=daemon_raw["server_port"],
        server_startup_timeout=daemon_raw["server_startup_timeout"],
        server_health_check_interval=daemon_raw["server_health_check_interval"],
    )

    # Transcription
    trans_raw = raw["transcription"]
    transcription = TranscriptionConfig(
        temperature=trans_raw["temperature"],
        temperature_increment=trans_raw["temperature_increment"],
        cli_timeout=trans_raw["cli_timeout"],
        server_timeout=trans_raw["server_timeout"],
        strip_leading_artifacts=trans_raw["strip_leading_artifacts"],
        strip_patterns=trans_raw["strip_patterns"],
        response_format=trans_raw["response_format"],
    )

    # Streaming
    stream_raw = raw["streaming"]
    debug_raw = stream_raw.get("debug", {})
    streaming_debug = StreamingDebugConfig(
        log_file=debug_raw.get("log_file", "/tmp/whisper_stream_debug.log"),
        output_log=debug_raw.get("output_log", "/tmp/whisper_stream_output.log"),
        stream_log=debug_raw.get("stream_log", "/tmp/whisper_stream.log"),
    )
    streaming = StreamingConfig(
        step=stream_raw["step"],
        buffer_length=stream_raw["buffer_length"],
        keep=stream_raw["keep"],
        vad_threshold=stream_raw["vad_threshold"],
        threads=stream_raw["threads"],
        max_committed_length=stream_raw["max_committed_length"],
        min_overlap_chars=stream_raw["min_overlap_chars"],
        overlap_step=stream_raw["overlap_step"],
        fallback_suffix_length=stream_raw["fallback_suffix_length"],
        drift_reset_threshold=stream_raw["drift_reset_threshold"],
        max_fallback_sentence_length=stream_raw["max_fallback_sentence_length"],
        noise_filter_pattern=stream_raw["noise_filter_pattern"],
        debug=streaming_debug,
    )

    # Wayland
    wayland_raw = raw["wayland"]
    xdg = wayland_raw["xdg_runtime_dir"]
    if xdg == "auto":
        xdg = f"/run/user/{os.getuid()}"
    wayland = WaylandConfig(
        display=wayland_raw["display"],
        xdg_runtime_dir=xdg,
        typer=wayland_raw["typer"],
        notifier=wayland_raw["notifier"],
        notification_timeout=wayland_raw["notification_timeout"],
    )

    # Waybar
    waybar_raw = raw["waybar"]
    icons_raw = waybar_raw.get("icons", {})
    icons = WaybarIconsConfig(
        cli_ready=icons_raw.get("cli_ready", "~"),
        cli_recording=icons_raw.get("cli_recording", "● dictation"),
        cli_processing=icons_raw.get("cli_processing", "dictation"),
        server_ready=icons_raw.get("server_ready", "◆"),
        server_recording=icons_raw.get("server_recording", "◆ dictation"),
        server_processing=icons_raw.get("server_processing", "dictation"),
        streaming=icons_raw.get("streaming", "~ streaming"),
        error=icons_raw.get("error", "~"),
    )
    waybar = WaybarConfig(
        update_interval=waybar_raw.get("update_interval", 0.5),
        icons=icons,
    )

    # Build
    build_raw = raw.get("build", {})
    build = BuildConfig(
        build_type=build_raw.get("build_type", "Release"),
        build_server=build_raw.get("build_server", True),
        build_sdl2=build_raw.get("build_sdl2", True),
    )

    return Config(
        paths=paths,
        model=model,
        backend=backend,
        audio=audio,
        daemon=daemon,
        transcription=transcription,
        streaming=streaming,
        wayland=wayland,
        waybar=waybar,
        build=build,
    )

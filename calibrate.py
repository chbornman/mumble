#!/usr/bin/env python3
"""
Calibrate stream-mode dedup knobs to a specific voice.

One-shot setup per machine: read a known passage at your natural cadence,
sweep the dedup config, pick values that minimize dropped and duplicated
words on YOUR audio. Respects your rhythm instead of forcing you to
adjust how you speak.

Interface:
    python calibrate.py                         # interactive record + sweep
    python calibrate.py --passage PATH          # use a specific passage
    python calibrate.py --audio WAV             # skip recording, reuse WAV
    python calibrate.py --report                # don't write config.local.toml
    python calibrate.py --apply                 # write tuned values (default)

Coverage:
  Swept:     streaming.keep, streaming.min_overlap_chars,
             streaming.overlap_step, streaming.fallback_suffix_length,
             streaming.drift_reset_threshold
  Not swept: streaming.vad_threshold — VAD is a live-audio property;
             simulating it offline requires a heavy voice-activity model.
             Tune it separately if stream-mode triggers early/late for you.

Never swept: values are written to config.local.toml; config.toml is
untouched so defaults for other users stay stable.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# sounddevice / scipy are already required for the daemon, so import the
# recording dependency at the top. scipy/wavfile is only needed when we
# synthesize WAVs from recorded chunks.
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

from benchmark import compute_wer
from config_loader import Config, load_config
# StreamDeduplicator lives in whisper_daemon; stream_dedup.py is a runtime
# wrapper that re-exports it and pulls in the whole daemon on import, so we
# go straight to the source class here.
from whisper_daemon import StreamDeduplicator


DEFAULT_SWEEPS: dict[str, list[Any]] = {
    # Streaming buffer overlap (ms). whisper-stream's rolling buffer keeps
    # the last `keep` ms across blocks; too little → dedup misses the
    # overlap; too much → redundant transcription.
    "keep": [100, 200, 300],
    "min_overlap_chars": [10, 15, 20, 25],
    "overlap_step": [5, 10, 15],
    "fallback_suffix_length": [30, 50, 70],
    "drift_reset_threshold": [2, 3],
}


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def compute_dupe_rate(hypothesis: str, reference: str) -> float:
    """Fraction of emitted words that are (a) not in the reference AND
    (b) a repeat of a word already emitted.

    Motivation: pure WER hides stream-mode's signature failure —
    "Inconsistent. Inconsistent." — because both instances are technically
    present in the reference sentence. Dupe-rate counts the second one
    as a problem only when it isn't justified by the ground truth.

    For words that appear K times in the reference, the first K emissions
    are allowed; the (K+1)th onward counts as a dupe.
    """
    hyp = _normalize_words(hypothesis)
    if not hyp:
        return 0.0
    ref_counts: dict[str, int] = {}
    for w in _normalize_words(reference):
        ref_counts[w] = ref_counts.get(w, 0) + 1

    seen: dict[str, int] = {}
    dupes = 0
    for w in hyp:
        seen[w] = seen.get(w, 0) + 1
        allowed = ref_counts.get(w, 0)
        if seen[w] > allowed and seen[w] > 1:
            dupes += 1
    return dupes / len(hyp)


@dataclass
class SweepResult:
    params: dict[str, Any]
    wer: float
    dupe_rate: float
    combined_text: str

    @property
    def score(self) -> float:
        """Lower is better. WER dominates; dupes are an additive penalty."""
        return self.wer + 0.5 * self.dupe_rate


@dataclass
class CalibrationReport:
    passage_name: str
    wav_path: Path
    window_ms: int
    ranked: list[SweepResult]
    per_knob_sensitivity: dict[str, list[tuple[Any, float]]] = field(
        default_factory=dict
    )


def slice_wav_windows(
    wav_path: Path,
    buffer_ms: int,
    keep_ms: int,
) -> list[Path]:
    """Produce sliding window WAVs emulating whisper-stream's rolling buffer.

    Each window is `buffer_ms` long. Stride is `buffer_ms - keep_ms`, so
    consecutive windows overlap by `keep_ms` — matching whisper-stream's
    --keep semantics.
    """
    if buffer_ms <= keep_ms:
        raise ValueError("buffer_ms must exceed keep_ms for a positive stride")
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(frames, dtype=dtype)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    window_samples = int(sample_rate * buffer_ms / 1000)
    stride_samples = int(sample_rate * (buffer_ms - keep_ms) / 1000)

    out_paths: list[Path] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="mumble-calibrate-"))
    start = 0
    idx = 0
    total = len(audio)
    while start < total:
        end = min(start + window_samples, total)
        chunk = audio[start:end]
        if len(chunk) < sample_rate // 4:
            # skip sub-250ms tails
            break
        out = tmpdir / f"window_{idx:04d}.wav"
        wavfile.write(out, sample_rate, chunk)
        out_paths.append(out)
        idx += 1
        start += stride_samples
    return out_paths


def _strip_whisper_artifacts(text: str) -> str:
    text = re.sub(r"^--\s*", "", text)
    text = re.sub(r"^-\s+", "", text)
    return text.strip()


def transcribe_window(
    whisper_cli: Path,
    model_path: Path,
    audio_file: Path,
    threads: int,
    language: str,
) -> str:
    """Transcribe a single audio window via whisper-cli. Returns text
    (empty on failure). Matches whisper_daemon.py's invocation so the
    transcript distribution mirrors what stream-mode actually produces."""
    cmd = [
        str(whisper_cli),
        "-m",
        str(model_path),
        "-f",
        str(audio_file),
        "-nt",
        "--no-prints",
        "-t",
        str(threads),
        "-l",
        language,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return ""
    lines = result.stdout.strip().splitlines()
    text_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith("whisper_")
        and not line.startswith("system_info")
        and not line.startswith("main:")
    ]
    return _strip_whisper_artifacts(" ".join(text_lines))


def _make_dedup(config: Config, overrides: dict[str, Any]) -> StreamDeduplicator:
    """Clone the streaming config with overrides and instantiate a fresh
    StreamDeduplicator. Kept local so sweep iterations never share state."""
    from copy import deepcopy

    cfg = deepcopy(config)
    for k, v in overrides.items():
        setattr(cfg.streaming, k, v)
    logger = logging.getLogger("calibrate.dedup")
    logger.setLevel(logging.WARNING)
    dedup = StreamDeduplicator(cfg, logger)
    return dedup


def simulate_dedup(
    partials: list[str],
    config: Config,
    overrides: dict[str, Any],
) -> str:
    """Drive StreamDeduplicator with a precomputed list of window transcripts.

    Returns the concatenated emitted text — the stream-mode equivalent of
    what the user would see typed into their editor.
    """
    dedup = _make_dedup(config, overrides)
    parts: list[str] = []
    for full_text in partials:
        if not full_text:
            continue
        new_text = dedup.extract_new_text(full_text)
        if new_text:
            dedup.commit(new_text)
            parts.append(new_text)
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def record_passage(
    sample_rate: int,
    channels: int,
    out_path: Path,
    min_seconds: int = 5,
) -> None:
    """Interactive: prompt the user to read, capture audio, write WAV.

    Blocks on stdin — the user presses Enter to start and Enter to stop
    so the whole flow stays visible in the terminal they invoked the
    subcommand from.
    """
    print("Press Enter when ready, speak the passage, then press Enter again to stop.")
    try:
        input()
    except EOFError:
        print("No TTY — run with --audio PATH to skip recording.", file=sys.stderr)
        sys.exit(2)

    recorded: list[np.ndarray] = []
    recording = True

    def callback(indata, frames, time_info, status):
        if recording:
            recorded.append(indata.copy())

    started = time.perf_counter()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        callback=callback,
        dtype="int16",
    ):
        print("Recording — press Enter to stop.")
        try:
            input()
        except EOFError:
            pass
        recording = False

    duration = time.perf_counter() - started
    if duration < min_seconds:
        print(
            f"Recording too short ({duration:.1f}s < {min_seconds}s); aborting.",
            file=sys.stderr,
        )
        sys.exit(2)
    audio = np.concatenate(recorded, axis=0) if recorded else np.zeros(0, np.int16)
    wavfile.write(out_path, sample_rate, audio)
    print(f"Recorded {duration:.1f}s → {out_path}")


def write_config_local_overrides(
    project_dir: Path, overrides: dict[str, Any]
) -> Path:
    """Append/merge a [streaming] override block into config.local.toml.

    Never touches config.toml. We rewrite only the streaming section's
    calibrated keys — existing non-streaming overrides are preserved.
    """
    local_path = project_dir / "config.local.toml"
    existing = ""
    if local_path.exists():
        existing = local_path.read_text()

    # Drop any existing calibrated keys inside an existing [streaming] section
    # so a rerun refreshes instead of stacking duplicates.
    pattern_keys = "|".join(re.escape(k) for k in overrides.keys())
    section_header_re = re.compile(r"^\s*\[(?P<name>[^\]]+)\]")
    stripped_lines: list[str] = []
    in_streaming = False
    has_streaming = False
    for line in existing.splitlines():
        header = section_header_re.match(line)
        if header:
            name = header.group("name").strip()
            in_streaming = name == "streaming"
            if in_streaming:
                has_streaming = True
            stripped_lines.append(line)
            continue
        if in_streaming and re.match(rf"^\s*({pattern_keys})\s*=", line):
            continue
        stripped_lines.append(line)
    merged = "\n".join(stripped_lines).rstrip()

    assignment_lines = [f"{k} = {v}  # calibrate.py" for k, v in overrides.items()]

    if has_streaming:
        # Inject the new keys directly after the existing [streaming] header
        # so the section stays contiguous and we don't create a duplicate.
        # Rebuild line-by-line: after we encounter the [streaming] header,
        # emit the new assignments immediately (keys we wanted to replace
        # were already stripped in the first pass above).
        rebuilt: list[str] = []
        inserted = False
        for line in merged.splitlines():
            rebuilt.append(line)
            header = section_header_re.match(line)
            if (
                not inserted
                and header
                and header.group("name").strip() == "streaming"
            ):
                rebuilt.extend(assignment_lines)
                inserted = True
        new_text = "\n".join(rebuilt)
    else:
        block = "\n".join(["", "[streaming]  # written by calibrate.py", *assignment_lines])
        new_text = (merged + "\n" + block + "\n").lstrip()

    local_path.write_text(new_text.rstrip() + "\n")
    return local_path


def run_sweep(
    config: Config,
    wav_path: Path,
    reference: str,
    buffer_ms: int,
    keep_values: list[int],
    dedup_sweeps: dict[str, list[Any]],
    max_combos: int | None = None,
) -> CalibrationReport:
    """Core sweep. Transcribes once per `keep` value (expensive) and reuses
    those partials for every dedup knob combination (cheap)."""
    ranked: list[SweepResult] = []
    knob_names = [k for k in dedup_sweeps if k != "keep"]
    knob_values = [dedup_sweeps[k] for k in knob_names]
    combos = list(itertools.product(*knob_values)) if knob_values else [()]
    if max_combos is not None:
        combos = combos[:max_combos]

    threads = config.backend.effective_threads
    language = config.model.language or "en"

    for keep_val in keep_values:
        print(f"\n[calibrate] keep={keep_val}ms — slicing WAV and transcribing windows")
        windows = slice_wav_windows(wav_path, buffer_ms, keep_val)
        partials = [
            transcribe_window(
                config.whisper_cli_path,
                config.model_path,
                w,
                threads,
                language,
            )
            for w in windows
        ]
        print(f"[calibrate] {len(partials)} windows transcribed")

        for combo in combos:
            overrides: dict[str, Any] = {"keep": keep_val}
            for name, value in zip(knob_names, combo):
                overrides[name] = value
            output = simulate_dedup(partials, config, overrides)
            wer = compute_wer(reference, output)
            dr = compute_dupe_rate(output, reference)
            ranked.append(
                SweepResult(
                    params=overrides,
                    wer=wer,
                    dupe_rate=dr,
                    combined_text=output,
                )
            )

    ranked.sort(key=lambda r: r.score)

    # Per-knob sensitivity: for each knob, average score across combos at each value.
    sensitivity: dict[str, list[tuple[Any, float]]] = {}
    for knob in ["keep"] + knob_names:
        values = sorted(set(r.params.get(knob) for r in ranked))
        buckets: list[tuple[Any, float]] = []
        for v in values:
            scores = [r.score for r in ranked if r.params.get(knob) == v]
            if scores:
                buckets.append((v, sum(scores) / len(scores)))
        sensitivity[knob] = buckets

    return CalibrationReport(
        passage_name=wav_path.stem,
        wav_path=wav_path,
        window_ms=buffer_ms,
        ranked=ranked,
        per_knob_sensitivity=sensitivity,
    )


def print_report(report: CalibrationReport) -> None:
    print("\n" + "=" * 72)
    print(f"Calibration report — {report.passage_name}")
    print("=" * 72)
    print("\nTop configurations (lower is better):")
    print(
        f"  {'#':<3} {'score':>8} {'WER':>7} {'dupe':>7}  params"
    )
    for i, r in enumerate(report.ranked[:5], start=1):
        params = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(
            f"  {i:<3} {r.score:>8.4f} {r.wer:>6.1%} {r.dupe_rate:>6.1%}  {params}"
        )

    print("\nPer-knob sensitivity (avg score by value — lower is better):")
    for knob, buckets in report.per_knob_sensitivity.items():
        summary = ", ".join(f"{v}:{score:.3f}" for v, score in buckets)
        print(f"  {knob:<28} {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate streaming dedup knobs to your voice."
    )
    parser.add_argument("--config", "-c", help="Path to config.toml")
    parser.add_argument(
        "--passage",
        help="Ground-truth text file (default: benchmarks/passages/technical.txt)",
    )
    parser.add_argument(
        "--audio",
        help="Skip recording; use this WAV as the calibration sample",
    )
    parser.add_argument(
        "--buffer-ms",
        type=int,
        default=5000,
        help="Simulated streaming block length (default: 5000ms)",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Cap dedup-knob combos per keep value (for quick iteration)",
    )
    apply_grp = parser.add_mutually_exclusive_group()
    apply_grp.add_argument(
        "--apply",
        dest="apply_mode",
        action="store_true",
        help="Write tuned values to config.local.toml (default)",
    )
    apply_grp.add_argument(
        "--report",
        dest="apply_mode",
        action="store_false",
        help="Print report only; do not modify config.local.toml",
    )
    parser.set_defaults(apply_mode=True)

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    project_dir = config.paths.project_dir
    passage_path = (
        Path(args.passage)
        if args.passage
        else project_dir / "benchmarks" / "passages" / "technical.txt"
    )
    if not passage_path.exists():
        print(f"Passage not found: {passage_path}", file=sys.stderr)
        sys.exit(1)
    reference = passage_path.read_text()
    # Strip the leading instructional header if present (everything before
    # the first blank line that itself precedes substantive text).
    body = reference.strip()
    # Heuristic: drop lines that are clearly guidance ("Read this aloud...").
    body_lines = body.splitlines()
    trimmed_lines = [
        ln
        for ln in body_lines
        if ln.strip() and not ln.strip().startswith("Calibration passage")
    ]
    reference = " ".join(trimmed_lines).strip()

    if args.audio:
        wav_path = Path(args.audio).expanduser()
        if not wav_path.exists():
            print(f"Audio not found: {wav_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Passage: {passage_path}")
        print("\n--- PASSAGE ---")
        print(passage_path.read_text().strip())
        print("--- END PASSAGE ---\n")
        tmpdir = Path(tempfile.mkdtemp(prefix="mumble-calibrate-rec-"))
        wav_path = tmpdir / "passage.wav"
        record_passage(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            out_path=wav_path,
        )

    if not config.whisper_cli_path.exists():
        print(
            f"whisper-cli not found at {config.whisper_cli_path} — build it first.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not config.model_path.exists():
        print(f"Model not found at {config.model_path}", file=sys.stderr)
        sys.exit(1)

    sweeps = dict(DEFAULT_SWEEPS)
    keep_values = sweeps.pop("keep")
    report = run_sweep(
        config,
        wav_path,
        reference,
        buffer_ms=args.buffer_ms,
        keep_values=keep_values,
        dedup_sweeps=sweeps,
        max_combos=args.max_combos,
    )

    print_report(report)

    if not report.ranked:
        print("No sweep results.", file=sys.stderr)
        sys.exit(1)

    best = report.ranked[0]
    if args.apply_mode:
        out_path = write_config_local_overrides(project_dir, best.params)
        print(f"\n[calibrate] Wrote best params to {out_path}")
    else:
        print("\n[calibrate] --report mode; config.local.toml unchanged.")


if __name__ == "__main__":
    main()

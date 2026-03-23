#!/usr/bin/env python3
"""
Benchmark script for comparing whisper.cpp configurations.

Tests different combinations of:
- Models (base.en, base.en-q5_1, small.en, etc.)
- Backends (cpu, vulkan)
- Thread counts (1, 2, 4, 8, 12, 16)

Uses pre-recorded audio files and compares transcription accuracy
against known reference texts using Word Error Rate (WER).

Usage:
    # Run all benchmarks with available models/audio
    python benchmark.py

    # Run with specific audio files
    python benchmark.py --audio-dir benchmarks/audio

    # Only test specific models
    python benchmark.py --models base.en base.en-q5_1 small.en

    # Only test specific thread counts
    python benchmark.py --threads 1 4 8 12

    # Quick mode: only test current config
    python benchmark.py --quick

    # Output results as JSON
    python benchmark.py --json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from config_loader import load_config


@dataclass
class BenchmarkResult:
    model: str
    backend: str
    threads: int
    audio_file: str
    audio_duration_s: float
    transcription_time_s: float
    realtime_factor: float  # audio_duration / transcription_time (higher = faster)
    transcription: str
    reference: str
    wer: float  # Word Error Rate (0.0 = perfect)
    cer: float  # Character Error Rate


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using dynamic programming (Levenshtein on words)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # DP table
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()

    if not ref:
        return 0.0 if not hyp else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref)][len(hyp)] / len(ref)


def get_audio_duration(audio_file: Path) -> float:
    """Get audio file duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def transcribe(
    whisper_cli: Path,
    model_path: Path,
    audio_file: Path,
    threads: int,
    vocab_prompt: str | None = None,
) -> tuple[str, float]:
    """Run whisper-cli and return (transcription, time_seconds)."""
    cmd = [
        str(whisper_cli),
        "-m",
        str(model_path),
        "-f",
        str(audio_file),
        "-nt",  # no timestamps
        "--no-prints",
        "-t",
        str(threads),
    ]

    if vocab_prompt:
        cmd.extend(["--prompt", vocab_prompt])

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        return f"[ERROR: {result.stderr.strip()[:200]}]", elapsed

    lines = result.stdout.strip().split("\n")
    text_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith("whisper_")
        and not line.startswith("system_info")
        and not line.startswith("main:")
    ]
    text = " ".join(text_lines).strip()

    # Strip leading artifacts
    import re

    text = re.sub(r"^--\s*", "", text)
    text = re.sub(r"^-\s+", "", text)

    return text, elapsed


def load_reference_texts(benchmarks_dir: Path) -> dict[str, str]:
    """Load reference texts from .txt files matching audio filenames."""
    refs = {}
    for txt_file in benchmarks_dir.glob("*.txt"):
        # Match txt to audio: passage_1.txt -> passage_1.wav or passage_1.mp3
        stem = txt_file.stem
        refs[stem] = txt_file.read_text().strip()
    return refs


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []
    for f in sorted(audio_dir.iterdir()):
        if f.suffix.lower() in extensions:
            files.append(f)
    return files


def find_available_models(models_dir: Path) -> list[str]:
    """Find all downloaded models."""
    models = []
    for f in sorted(models_dir.iterdir()):
        if f.name.startswith("ggml-") and f.name.endswith(".bin"):
            # Skip test models
            if f.name.startswith("for-tests-"):
                continue
            name = f.name[5:-4]  # strip ggml- and .bin
            models.append(name)
    return models


def run_benchmarks(
    config,
    audio_files: list[Path],
    references: dict[str, str],
    models: list[str],
    thread_counts: list[int],
    backends: list[str],
) -> list[BenchmarkResult]:
    """Run all benchmark combinations."""
    results = []
    whisper_cli = config.whisper_cli_path
    models_dir = config.paths.models_dir

    # Load vocab
    vocab_prompt = None
    if config.paths.vocab_file and config.paths.vocab_file.exists():
        with open(config.paths.vocab_file) as f:
            words = []
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    words.extend(w.strip() for w in line.split(",") if w.strip())
            vocab_prompt = ", ".join(words)

    total = len(audio_files) * len(models) * len(thread_counts)
    current = 0

    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)
        ref = references.get(audio_file.stem, "")

        for model_name in models:
            model_path = models_dir / f"ggml-{model_name}.bin"
            if not model_path.exists():
                continue

            for threads in thread_counts:
                current += 1
                backend = config.backend.type  # Use current build's backend

                print(
                    f"  [{current}/{total}] {model_name} | "
                    f"threads={threads} | {audio_file.name}...",
                    end="",
                    flush=True,
                )

                text, elapsed = transcribe(
                    whisper_cli, model_path, audio_file, threads, vocab_prompt
                )

                rtf = duration / elapsed if elapsed > 0 else 0
                wer = compute_wer(ref, text) if ref else -1.0
                cer = compute_cer(ref, text) if ref else -1.0

                result = BenchmarkResult(
                    model=model_name,
                    backend=backend,
                    threads=threads,
                    audio_file=audio_file.name,
                    audio_duration_s=round(duration, 2),
                    transcription_time_s=round(elapsed, 3),
                    realtime_factor=round(rtf, 2),
                    transcription=text,
                    reference=ref,
                    wer=round(wer, 4),
                    cer=round(cer, 4),
                )
                results.append(result)

                wer_str = f"WER={wer:.1%}" if ref else "no ref"
                print(f" {elapsed:.2f}s ({rtf:.1f}x realtime, {wer_str})")

    return results


def print_summary_table(results: list[BenchmarkResult]):
    """Print a formatted summary table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Group by audio file
    audio_files = sorted(set(r.audio_file for r in results))

    for audio in audio_files:
        audio_results = [r for r in results if r.audio_file == audio]
        duration = audio_results[0].audio_duration_s

        print(f"\n--- {audio} ({duration:.1f}s) ---")
        print(
            f"{'Model':<22} {'Backend':<8} {'Threads':>7} "
            f"{'Time':>8} {'RTF':>6} {'WER':>8} {'CER':>8}"
        )
        print("-" * 80)

        for r in sorted(audio_results, key=lambda x: x.transcription_time_s):
            wer_str = f"{r.wer:.1%}" if r.wer >= 0 else "n/a"
            cer_str = f"{r.cer:.1%}" if r.cer >= 0 else "n/a"
            print(
                f"{r.model:<22} {r.backend:<8} {r.threads:>7} "
                f"{r.transcription_time_s:>7.2f}s {r.realtime_factor:>5.1f}x "
                f"{wer_str:>8} {cer_str:>8}"
            )

    # Best configurations
    print("\n" + "=" * 100)
    print("BEST CONFIGURATIONS")
    print("=" * 100)

    valid = [r for r in results if r.wer >= 0]
    if valid:
        fastest = min(valid, key=lambda r: r.transcription_time_s)
        most_accurate = min(valid, key=lambda r: r.wer)
        best_balanced = min(
            valid, key=lambda r: r.transcription_time_s * (1 + r.wer * 5)
        )

        print(
            f"\n  Fastest:       {fastest.model} | {fastest.threads}t | "
            f"{fastest.transcription_time_s:.2f}s ({fastest.realtime_factor:.1f}x) | "
            f"WER={fastest.wer:.1%}"
        )
        print(
            f"  Most accurate: {most_accurate.model} | {most_accurate.threads}t | "
            f"{most_accurate.transcription_time_s:.2f}s ({most_accurate.realtime_factor:.1f}x) | "
            f"WER={most_accurate.wer:.1%}"
        )
        print(
            f"  Best balanced: {best_balanced.model} | {best_balanced.threads}t | "
            f"{best_balanced.transcription_time_s:.2f}s ({best_balanced.realtime_factor:.1f}x) | "
            f"WER={best_balanced.wer:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark whisper.cpp configurations")
    parser.add_argument("--config", "-c", help="Path to config.toml")
    parser.add_argument(
        "--audio-dir",
        help="Directory with audio files (default: benchmarks/audio)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to test (default: all available)",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        help="Thread counts to test (default: 1 2 4 8 12)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only test current model with a few thread counts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to file",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Find audio files
    project_dir = config.paths.project_dir
    audio_dir = (
        Path(args.audio_dir) if args.audio_dir else project_dir / "benchmarks" / "audio"
    )
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        print(f"Create it and add audio files, or use --audio-dir")
        print(f"\nExpected structure:")
        print(f"  {audio_dir}/")
        print(f"    passage_1.wav     # Your recorded audio")
        print(f"    passage_1.txt     # Reference text for that audio")
        print(f"    passage_2.wav")
        print(f"    passage_2.txt")
        sys.exit(1)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print("Supported formats: .wav, .mp3, .flac, .ogg, .m4a")
        sys.exit(1)

    # Load reference texts
    references = load_reference_texts(audio_dir)

    # Determine models to test
    if args.models:
        models = args.models
    elif args.quick:
        models = [config.model.name]
    else:
        models = find_available_models(config.paths.models_dir)
        if not models:
            print(f"No models found in {config.paths.models_dir}")
            sys.exit(1)

    # Determine thread counts
    if args.threads:
        thread_counts = args.threads
    elif args.quick:
        thread_counts = [4, 8, config.backend.effective_threads]
        thread_counts = sorted(set(thread_counts))
    else:
        thread_counts = [1, 2, 4, 8, 12]

    backends = [config.backend.type]

    # Print header
    print("=" * 60)
    print("Whisper Benchmark")
    print("=" * 60)
    print(f"  Backend:      {config.backend.type}")
    print(f"  Models:       {', '.join(models)}")
    print(f"  Threads:      {', '.join(str(t) for t in thread_counts)}")
    print(f"  Audio files:  {len(audio_files)}")
    print(f"  Ref texts:    {len(references)}")
    print(f"  Total runs:   {len(audio_files) * len(models) * len(thread_counts)}")
    print("=" * 60)
    print()

    # Verify whisper-cli exists
    if not config.whisper_cli_path.exists():
        print(f"Error: whisper-cli not found at {config.whisper_cli_path}")
        print("Build it first with: ./build_whisper.sh")
        sys.exit(1)

    # Run benchmarks
    results = run_benchmarks(
        config, audio_files, references, models, thread_counts, backends
    )

    # Print summary
    if not args.json:
        print_summary_table(results)

    # Output
    if args.json or args.output:
        data = [asdict(r) for r in results]
        json_str = json.dumps(data, indent=2)

        if args.json:
            print(json_str)

        if args.output:
            Path(args.output).write_text(json_str)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

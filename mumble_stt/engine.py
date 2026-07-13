"""NeMo cache-aware streaming engine for the mumble-stt sidecar.

This is the ONLY module that imports torch / NeMo. Keeping the heavy deps here
means ``server.py`` stays importable under any interpreter for ``--selftest``
probes; the engine is constructed lazily so its imports only fire when a real
model load is requested.

Wraps ``nvidia/nemotron-speech-streaming-en-0.6b`` (Cache-Aware FastConformer
RNNT) via NeMo 2.7's high-level cache-aware inference pipeline. The pipeline
owns the rolling encoder cache, the stateful RNNT decode (a partial Hypothesis
carried frame-to-frame), the RNNT EOU endpointer, and the native
punctuation+capitalization text processor. We feed it raw audio one native
chunk (160 ms) at a time and read back partial vs final per chunk.

Lifecycle the engine exposes to the server:
    load()                  -> import NeMo, build pipeline, warm up (once)
    open_utterance()        -> start a fresh streaming session (first chunk)
    feed(pcm_bytes)         -> buffer PCM, run whole 160ms chunks, yield events
    finalize()              -> flush the trailing partial as a final (FLUSH/BYE)
    close_utterance()       -> tear down per-utterance state

The verified decoder gotcha is encoded in :func:`_build_cfg`: strategy
``greedy_batch`` + ``greedy.loop_labels=True`` (anything else crashes/desyncs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

log = logging.getLogger("mumble_stt.engine")


@dataclass
class TranscriptEvent:
    """One incremental result from feeding a chunk.

    ``kind`` is "partial" or "final". For "final", ``text`` is the committed,
    punctuated+capitalized segment and the engine has reset its per-utterance
    hypothesis (utt counter handled by the server). For "partial", ``text`` is
    the growing, replaceable hypothesis since the last EOU.
    """
    kind: str   # "partial" | "final"
    text: str


class EngineError(RuntimeError):
    """Raised for an inference-level failure (transient) the server can map to
    an ERROR{code="inference"} and recover from via a state reset."""


class StreamingEngine:
    """Holds the NeMo model resident and runs cache-aware streaming inference."""

    def __init__(self, model_name: str, att_context_size: list[int],
                 sample_rate: int = 16000, eou_silence_ms: int = 800,
                 device: str = "cuda", vocab_phrases: Optional[list[str]] = None,
                 vocab_biasing_context_score: float = 1.0,
                 vocab_biasing_depth_scaling: float = 2.0,
                 vocab_biasing_alpha: float = 1.0):
        self.model_name = model_name
        self.att_context_size = att_context_size
        self.sample_rate = sample_rate
        self.eou_silence_ms = eou_silence_ms
        self.requested_device = device
        self.vocab_phrases = list(vocab_phrases or [])
        self.vocab_biasing_context_score = vocab_biasing_context_score
        self.vocab_biasing_depth_scaling = vocab_biasing_depth_scaling
        self.vocab_biasing_alpha = vocab_biasing_alpha

        # Set during load().
        self._pipeline = None
        self._device = None
        self._Frame = None
        self._ASRRequestOptions = None
        self._torch = None
        self._np = None
        self._chunk_samples = 0          # samples per native step (e.g. 2560)
        self._loaded = False

        # Per-utterance state.
        self._session_open = False
        self._first_chunk_pending = True
        self._pcm_buffer = bytearray()   # leftover bytes (< one full chunk)
        self._last_partial = ""
        self._stream_id = 0

    # --- model load / warmup ------------------------------------------------
    def load(self) -> None:
        """Import NeMo, build the pipeline once, and warm the inference backend.

        Raises on failure; the server maps that to ERROR{code="model_load"}
        and exits non-zero so systemd Restart kicks in.
        """
        if self._loaded:
            return

        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )

        self._np = np
        self._torch = torch
        self._Frame = Frame
        self._ASRRequestOptions = ASRRequestOptions

        log.info("building cache-aware pipeline for %s (att_context=%s)",
                 self.model_name, self.att_context_size)
        if self.vocab_phrases:
            log.info(
                "enabling RNNT vocabulary biasing with %d phrases "
                "(context_score=%g, depth_scaling=%g, alpha=%g, triton=%s)",
                len(self.vocab_phrases),
                self.vocab_biasing_context_score,
                self.vocab_biasing_depth_scaling,
                self.vocab_biasing_alpha,
                self.requested_device == "cuda",
            )
        cfg = OmegaConf.create(self._build_cfg())
        self._pipeline = PipelineBuilder.build_pipeline(cfg)
        self._device = self._pipeline.device

        # Feed EXACTLY the pipeline's native chunk per step (derived from the
        # model). Feeding less desyncs the RNNT token/timestamp bookkeeping.
        native_chunk_secs = self._pipeline.chunk_size_in_secs
        self._chunk_samples = int(round(native_chunk_secs * self.sample_rate))
        log.info("native chunk = %.0fms (%d samples/step), device=%s",
                 native_chunk_secs * 1000, self._chunk_samples, self._device)

        self._warmup()
        self._loaded = True
        log.info("engine loaded and warmed up")

    def _build_cfg(self) -> dict:
        """Assemble the DictConfig dict the cache-aware RNNT builder expects.

        Mirrors the verified stream_infer.py config. CRITICAL decoder settings:
        strategy='greedy_batch' + greedy.loop_labels=True. Do NOT set
        streaming.chunk_size_in_secs (lets the pipeline derive the native
        160ms chunk and keeps tokens_per_frame consistent).
        """
        greedy_cfg = {"max_symbols": 10, "loop_labels": True}
        if self.vocab_phrases:
            # Static vocabulary is decoder-wide, so the global fusion model is
            # a better fit than NeMo's per-request biasing hook (which is not
            # integrated by CacheAwareRNNTPipeline in NeMo 2.7). It naturally
            # survives EOU resets and applies to every frame. On CPU, forcing
            # the PyTorch implementation is essential: Triton can be installed
            # even when CUDA is unavailable.
            greedy_cfg.update({
                "use_cuda_graph_decoder": self.requested_device == "cuda",
                "boosting_tree": {
                    "key_phrases_list": self.vocab_phrases,
                    "context_score": self.vocab_biasing_context_score,
                    "depth_scaling": self.vocab_biasing_depth_scaling,
                    "use_triton": self.requested_device == "cuda",
                },
                "boosting_tree_alpha": self.vocab_biasing_alpha,
            })

        return {
            "pipeline_type": "cache_aware",
            "asr_decoding_type": "rnnt",
            "log_level": 30,            # WARNING (NeMo internal)
            "matmul_precision": "high",
            "cache_dir": None,
            "lang": "en",
            "enable_pnc": True,         # native punctuation + capitalization
            "enable_itn": False,        # no inverse text normalization dep
            "enable_nmt": False,        # no translation
            "asr_output_granularity": "word",
            "return_tail_result": True,  # flush trailing partial as final at end
            "asr": {
                "model_name": self.model_name,
                "device": self.requested_device,
                "device_id": 0 if self.requested_device == "cuda" else None,
                "compute_dtype": (
                    "bfloat16" if self.requested_device == "cuda" else "float32"
                ),
                "use_amp": False,
                "decoding": {
                    # The batched loop_labels greedy decoder restores BOTH
                    # y_sequence AND timestamp cumulatively via hyp.merge_();
                    # the non-batched path resets timestamp=[] each chunk and
                    # crashes with IndexError. Keep the batched path.
                    "strategy": "greedy_batch",
                    "greedy": greedy_cfg,
                },
            },
            "streaming": {
                "att_context_size": self.att_context_size,
                "sample_rate": self.sample_rate,
                # do NOT set chunk_size_in_secs (see docstring).
                "use_cache": True,
                "use_feat_cache": True,
                "batch_size": 1,
                "num_slots": 1,
                "request_type": "frame",
                "word_boundary_tolerance": 1,
            },
            "endpointing": {
                "stop_history_eou": self.eou_silence_ms,
                "residue_tokens_at_end": 4,
            },
            "confidence": {
                "method_cfg": {
                    "name": "max_prob",
                    "entropy_type": "tsallis",
                    "entropy_norm": "lin",
                    "alpha": 0.33,
                },
                "aggregation": "mean",
            },
            "itn": {
                "batch_size": 1,
                "n_jobs": 1,
                "left_padding_size": 0,
                "whitelist": None,
                "overwrite_cache": False,
                "max_number_of_permutations_per_split": 729,
            },
        }

    def _warmup(self) -> None:
        """Run one dummy silence frame so the first real chunk isn't skewed by
        backend kernel/graph warmup."""
        torch = self._torch
        n = self._chunk_samples
        self._pipeline.open_session()
        warm = torch.zeros(n, dtype=torch.float32, device=self._device)
        self._pipeline.transcribe_step([self._Frame(
            samples=warm, stream_id=999, is_first=True, is_last=True,
            length=n, options=self._ASRRequestOptions())])
        self._pipeline.close_session()

    # --- per-utterance streaming -------------------------------------------
    def open_utterance(self) -> None:
        """Begin a fresh streaming session for a new utterance."""
        if not self._loaded:
            raise EngineError("engine not loaded")
        if self._session_open:
            # Defensive: tear down a dangling session first.
            self.close_utterance()
        self._pipeline.open_session()
        self._session_open = True
        self._first_chunk_pending = True
        self._pcm_buffer = bytearray()
        self._last_partial = ""

    def feed(self, pcm: bytes) -> Iterator[TranscriptEvent]:
        """Buffer incoming PCM and run whole native chunks through the model.

        Yields a TranscriptEvent for each chunk whose output changed: a "final"
        when the endpointer fires EOU (state then auto-resets for the next
        utterance), otherwise a "partial" when the hypothesis grows. Leftover
        bytes (< one chunk) stay buffered for the next call.
        """
        if not self._session_open:
            raise EngineError("feed() before open_utterance()")
        self._pcm_buffer.extend(pcm)
        bytes_per_chunk = self._chunk_samples * 2  # int16 -> 2 bytes/sample

        while len(self._pcm_buffer) >= bytes_per_chunk:
            chunk_bytes = bytes(self._pcm_buffer[:bytes_per_chunk])
            del self._pcm_buffer[:bytes_per_chunk]
            ev = self._run_chunk(chunk_bytes, is_last=False)
            if ev is not None:
                yield ev

    def finalize(self) -> Iterator[TranscriptEvent]:
        """Flush on an utterance boundary (FLUSH/BYE or stream end).

        Pads and runs any buffered tail as the LAST frame (is_last=True), which
        fires a final EOU, runs the text processor on the tail, and tears down
        the session. Yields the trailing partial(s) then the final.
        """
        if not self._session_open:
            return

        bytes_per_chunk = self._chunk_samples * 2
        tail = bytes(self._pcm_buffer)
        self._pcm_buffer = bytearray()

        # Real (un-padded) sample count of the tail, captured BEFORE padding so
        # the Frame's length= reflects only valid audio. 0 valid is fine: a
        # silence-padded terminal frame still flushes the trailing partial.
        valid = min(len(tail) // 2, self._chunk_samples)
        if len(tail) < bytes_per_chunk:
            tail = tail + b"\x00" * (bytes_per_chunk - len(tail))

        try:
            ev = self._run_chunk(tail, is_last=True, valid_override=valid)
        finally:
            # The pipeline deletes per-stream state on the is_last frame; reset
            # our bookkeeping unconditionally so a decode error still ends the
            # session cleanly.
            self._session_open = False
            self._first_chunk_pending = True
            self._last_partial = ""
        if ev is not None:
            yield ev

    def close_utterance(self) -> None:
        """Tear down the streaming session without forcing a final (used on
        abrupt disconnect / error reset)."""
        if self._session_open and self._pipeline is not None:
            try:
                self._pipeline.close_session()
            except Exception:  # pragma: no cover - best-effort teardown
                log.debug("close_session during teardown raised", exc_info=True)
        self._session_open = False
        self._first_chunk_pending = True
        self._pcm_buffer = bytearray()
        self._last_partial = ""

    # --- internals ----------------------------------------------------------
    def _run_chunk(self, chunk_bytes: bytes, is_last: bool,
                   valid_override: Optional[int] = None) -> Optional[TranscriptEvent]:
        """Convert one chunk of s16le PCM to a Frame and run one transcribe_step.

        Returns a TranscriptEvent or None (no change). Maps exceptions to
        EngineError so the server can emit a non-fatal inference error.
        """
        np = self._np
        torch = self._torch

        audio_i16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        if len(audio_i16) != self._chunk_samples:
            # Pad/trim defensively so the tensor is always native length.
            if len(audio_i16) < self._chunk_samples:
                audio_i16 = np.pad(audio_i16, (0, self._chunk_samples - len(audio_i16)))
            else:
                audio_i16 = audio_i16[: self._chunk_samples]
        valid = valid_override if valid_override is not None else self._chunk_samples
        valid = max(0, min(valid, self._chunk_samples))

        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        samples = torch.from_numpy(audio_f32).to(self._device)

        is_first = self._first_chunk_pending
        self._first_chunk_pending = False

        frame = self._Frame(
            samples=samples, stream_id=self._stream_id,
            is_first=is_first, is_last=is_last,
            length=valid, options=self._ASRRequestOptions())

        try:
            outputs = self._pipeline.transcribe_step([frame])
        except Exception as e:
            raise EngineError(f"transcribe_step failed: {e}") from e

        out = outputs[0]
        final_text = out.final_transcript.strip() if out.final_transcript else ""
        if final_text:
            self._last_partial = ""
            return TranscriptEvent(kind="final", text=final_text)

        # lstrip: NeMo detokenizes partials "without stripping", so the leading
        # SentencePiece underscore becomes a leading space. Finals are stripped
        # (above), so an unstripped partial would diverge from its final at
        # char 0 and force the daemon's live prefix-diff to rewrite the whole
        # utterance on every EOU.
        partial_text = (out.partial_transcript or "").lstrip()
        if partial_text and partial_text != self._last_partial:
            self._last_partial = partial_text
            return TranscriptEvent(kind="partial", text=partial_text)
        return None

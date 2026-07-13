"""mumble-stt sidecar entrypoint.

Opens the unix socket, accepts ONE streaming session at a time, and drives the
load-once / reset-per-utterance / ERROR lifecycle described in the sidecar
protocol section of ``ARCHITECTURE.md``. All torch/NeMo work is confined to ``engine.py``;
this module imports it lazily so ``--selftest`` can run under any interpreter.

Run by hand for debugging (from the repo root):
    .venv-stt/bin/python mumble_stt/server.py --config config.toml

Probe without loading the model (light venv OK):
    python mumble_stt/server.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Make the sidecar package importable whether invoked as a file or a module.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_DIR = _THIS_DIR.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from mumble_stt import protocol as proto  # noqa: E402
from mumble_stt.config import SidecarConfig, load_sidecar_config  # noqa: E402

log = logging.getLogger("mumble_stt.server")

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class Sidecar:
    """Owns the socket, the resident engine, and the single-session gate."""

    def __init__(self, cfg: SidecarConfig):
        self.cfg = cfg
        self.engine = None              # lazily constructed in load_engine()
        self._session_active = False    # single-session gate
        self._server: asyncio.AbstractServer | None = None
        self._shutting_down = False

    # --- model load ---------------------------------------------------------
    def load_engine(self) -> None:
        """Construct and load the NeMo engine. Heavy imports happen here.

        On failure, the caller emits ERROR{model_load} once (no client yet, so
        it just logs) and exits non-zero so systemd Restart kicks in.
        """
        from mumble_stt.engine import StreamingEngine  # heavy import, lazy
        from mumble_stt.vocabulary import load_vocabulary

        vocab_phrases = []
        if self.cfg.vocab_biasing_enabled:
            vocab_phrases = load_vocabulary(
                self.cfg.vocab_file, self.cfg.vocab_biasing_max_phrases
            )
            if not vocab_phrases:
                raise ValueError(
                    f"vocabulary biasing enabled but {self.cfg.vocab_file} has no phrases"
                )

        self.engine = StreamingEngine(
            model_name=self.cfg.model,
            att_context_size=self.cfg.att_context_size,
            sample_rate=self.cfg.sample_rate,
            eou_silence_ms=self.cfg.eou_silence_ms,
            device=self.cfg.device,
            vocab_phrases=vocab_phrases,
            vocab_biasing_context_score=self.cfg.vocab_biasing_context_score,
            vocab_biasing_depth_scaling=self.cfg.vocab_biasing_depth_scaling,
            vocab_biasing_alpha=self.cfg.vocab_biasing_alpha,
        )
        self.engine.load()

    # --- socket lifecycle ---------------------------------------------------
    async def serve(self) -> None:
        sock_path = self.cfg.socket_path
        # Unlink a stale socket from a previous (crashed) run.
        try:
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        except OSError as e:
            log.warning("could not unlink stale socket %s: %s", sock_path, e)

        os.makedirs(os.path.dirname(sock_path), exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=sock_path)
        try:
            os.chmod(sock_path, 0o600)
        except OSError as e:
            log.warning("could not chmod socket 0600: %s", e)

        log.info("listening on %s", sock_path)
        async with self._server:
            await self._server.serve_forever()

    async def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        log.info("shutting down")
        if self._server is not None:
            self._server.close()
        # Best-effort socket cleanup.
        try:
            if os.path.exists(self.cfg.socket_path):
                os.unlink(self.cfg.socket_path)
        except OSError:
            pass

    # --- one connection -----------------------------------------------------
    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        peer = "client"
        # Single-session gate: a second connection while one is active gets an
        # immediate ERROR{busy} then close.
        if self._session_active:
            log.info("rejecting concurrent connection (busy)")
            await self._send(writer, proto.error(
                proto.ERR_BUSY, "another session is active", fatal=True))
            await self._close(writer)
            return

        self._session_active = True
        session = _Session(self.engine, self.cfg, writer)
        try:
            await session.run(reader)
        except proto.ProtocolError as e:
            log.info("%s: protocol error: %s", peer, e)
        except Exception as e:  # pragma: no cover - defensive
            log.exception("%s: unexpected session error: %s", peer, e)
            try:
                await self._send(writer, proto.error(
                    proto.ERR_INTERNAL, str(e), fatal=True))
            except Exception:
                pass
        finally:
            session.cleanup()
            await self._close(writer)
            self._session_active = False
            log.info("session ended; ready for next")

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, frame: bytes) -> None:
        writer.write(frame)
        await writer.drain()

    @staticmethod
    async def _close(writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


class _Session:
    """Processes one connection: HELLO -> READY -> (AUDIO/FLUSH)* -> BYE."""

    def __init__(self, engine, cfg: SidecarConfig, writer: asyncio.StreamWriter):
        self.engine = engine
        self.cfg = cfg
        self.writer = writer
        self.session_id = "?"
        self.want_partials = True
        self.utt = 0
        self._utt_open = False

    async def run(self, reader: asyncio.StreamReader) -> None:
        # 1. Expect HELLO first.
        type_tag, payload = await proto.read_frame_async(reader)
        if type_tag != proto.TYPE_HELLO:
            await self._send(proto.error(
                proto.ERR_INTERNAL,
                f"expected HELLO, got {proto.TYPE_NAMES.get(type_tag, type_tag)}",
                fatal=True))
            return

        hello = proto.decode_json(payload)
        if not await self._validate_hello(hello):
            return
        self.session_id = str(hello.get("session_id", "?"))
        self.want_partials = bool(hello.get("want_partials", True))

        # 2. Reset streaming state and reply READY.
        self._open_utterance()
        await self._send(proto.ready(
            self.cfg.model, self.cfg.sample_rate, self.cfg.att_context))
        log.info("session %s ready (want_partials=%s)",
                 self.session_id, self.want_partials)

        # 3. Main loop: AUDIO / FLUSH / BYE.
        while True:
            type_tag, payload = await proto.read_frame_async(reader)

            if type_tag == proto.TYPE_AUDIO:
                await self._on_audio(payload)
            elif type_tag == proto.TYPE_FLUSH:
                await self._on_flush()
            elif type_tag == proto.TYPE_BYE:
                await self._on_bye()
                return
            elif type_tag == proto.TYPE_HELLO:
                await self._send(proto.error(
                    proto.ERR_INTERNAL, "duplicate HELLO", fatal=False))
            else:
                log.debug("ignoring unexpected frame type 0x%02x", type_tag)

    # --- validation ---------------------------------------------------------
    async def _validate_hello(self, hello: dict) -> bool:
        sr = hello.get("sample_rate", self.cfg.sample_rate)
        enc = hello.get("encoding", "s16le")
        ch = hello.get("channels", 1)
        problems = []
        if sr != self.cfg.sample_rate:
            problems.append(f"sample_rate {sr} != {self.cfg.sample_rate}")
        if enc != "s16le":
            problems.append(f"encoding {enc!r} != 's16le'")
        if ch != 1:
            problems.append(f"channels {ch} != 1")
        if problems:
            msg = "; ".join(problems)
            log.info("rejecting HELLO: %s", msg)
            await self._send(proto.error(proto.ERR_BAD_AUDIO, msg, fatal=True))
            return False
        return True

    # --- per-message handlers ----------------------------------------------
    def _open_utterance(self) -> None:
        if self.engine is not None:
            self.engine.open_utterance()
        self._utt_open = True

    async def _on_audio(self, pcm: bytes) -> None:
        if not self._utt_open:
            self._open_utterance()
        if self.engine is None:
            return
        try:
            events = list(self.engine.feed(pcm))
        except Exception as e:  # EngineError -> non-fatal inference error
            await self._on_inference_error(e)
            return
        await self._emit(events)

    async def _on_flush(self) -> None:
        """Finalize the current utterance, reset state, stay connected."""
        await self._finalize_current()
        self.utt += 1
        self._open_utterance()

    async def _on_bye(self) -> None:
        """Finalize any in-flight utterance, then close cleanly."""
        await self._finalize_current()
        log.info("session %s: BYE", self.session_id)

    async def _finalize_current(self) -> None:
        if not self._utt_open or self.engine is None:
            self._utt_open = False
            return
        try:
            events = list(self.engine.finalize())
        except Exception as e:
            await self._on_inference_error(e)
            self._utt_open = False
            return
        self._utt_open = False
        await self._emit(events)

    async def _emit(self, events) -> None:
        """Send PARTIAL/FINAL frames for engine events, honoring want_partials."""
        for ev in events:
            if ev.kind == "final":
                await self._send(proto.final(self.utt, ev.text))
            elif ev.kind == "partial" and self.want_partials:
                await self._send(proto.partial(self.utt, ev.text))

    async def _on_inference_error(self, exc: Exception) -> None:
        """Transient decode failure: report non-fatal, reset utterance state."""
        log.warning("session %s: inference error: %s", self.session_id, exc)
        await self._send(proto.error(proto.ERR_INFERENCE, str(exc), fatal=False))
        if self.engine is not None:
            self.engine.close_utterance()
        self.utt += 1
        self._open_utterance()

    # --- io -----------------------------------------------------------------
    async def _send(self, frame: bytes) -> None:
        self.writer.write(frame)
        await self.writer.drain()

    def cleanup(self) -> None:
        """Tear down per-session engine state on disconnect (without forcing a
        final — an abrupt drop shouldn't inject anything)."""
        if self.engine is not None:
            self.engine.close_utterance()


# --- entrypoint -------------------------------------------------------------
def _configure_logging(level_name: str) -> None:
    level = _LOG_LEVELS.get(level_name.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _selftest(cfg: SidecarConfig) -> int:
    """Light, non-destructive probe: validate config + protocol round-trips
    without importing torch/NeMo. Returns a process exit code."""
    print(f"[selftest] socket_path   = {cfg.socket_path}")
    print(f"[selftest] model         = {cfg.model}")
    print(f"[selftest] sample_rate   = {cfg.sample_rate}")
    print(f"[selftest] venv_python   = {cfg.venv_python}")
    print(f"[selftest] att_context   = {cfg.att_context} -> {cfg.att_context_size}")
    print(f"[selftest] log_level     = {cfg.log_level}")
    print(f"[selftest] device        = {cfg.device} ({cfg.compute_dtype})")
    print(f"[selftest] vocab_biasing = {cfg.vocab_biasing_enabled}")
    print(f"[selftest] vocab_file    = {cfg.vocab_file}")

    # Protocol round-trip sanity (stdlib only).
    frame = proto.hello("sid-123", want_partials=True)
    t, p = proto._HEADER.unpack(frame[:5])
    obj = proto.decode_json(frame[5:])
    assert t == proto.TYPE_HELLO and obj["session_id"] == "sid-123", "hello roundtrip"
    f = proto.final(2, "Hello, world.")
    assert proto.decode_json(f[5:])["text"] == "Hello, world.", "final roundtrip"
    a = proto.audio(b"\x00\x01" * 1600)
    at, _ = proto._HEADER.unpack(a[:5])
    assert at == proto.TYPE_AUDIO and len(a[5:]) == 3200, "audio roundtrip"
    print("[selftest] protocol round-trips OK")

    venv_ok = os.access(cfg.venv_python, os.X_OK)
    print(f"[selftest] heavy venv python executable: {venv_ok}")
    print("[selftest] OK (model NOT loaded; use a full run to load NeMo)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mumble-stt Nemotron streaming sidecar")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--selftest", action="store_true",
                        help="validate config + protocol without loading NeMo")
    args = parser.parse_args(argv)

    cfg = load_sidecar_config(args.config)
    _configure_logging(cfg.log_level)

    if args.selftest:
        return _selftest(cfg)

    sidecar = Sidecar(cfg)

    # Load the model BEFORE accepting connections so the first READY is honest.
    try:
        sidecar.load_engine()
    except Exception as e:
        log.error("model load failed: %s", e, exc_info=True)
        # No client yet; emit a machine-readable line and exit non-zero so
        # systemd Restart kicks in.
        sys.stderr.write(
            proto.error(proto.ERR_MODEL_LOAD, str(e), fatal=True)[5:].decode("utf-8")
            + "\n")
        return 1

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, lambda: asyncio.ensure_future(sidecar.shutdown()))
            except NotImplementedError:  # pragma: no cover
                pass
        try:
            await sidecar.serve()
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

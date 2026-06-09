"""``nemotron-streaming`` backend — daemon-side client to the mumble-stt sidecar.

A thin, blocking unix-socket client that connects to the sidecar, streams
captured mic PCM in, and reads PARTIAL/FINAL/ERROR back. It imports ONLY
``mumble_stt.protocol`` (pure stdlib: struct, json, socket), so it loads fine
inside mumble's LIGHT venv — no torch/NeMo ever crosses the socket.

This module is a standalone, reusable reference client for the wire contract.
It is also handy for exercising the sidecar by hand (see ``__main__`` below).
The daemon (``whisper_daemon.py``) currently inlines an equivalent client; this
keeps the client form documented and testable in one importable place.

Typical use::

    client = NemotronStreamingClient(socket_path)
    client.connect(session_id="abc", want_partials=True)   # waits for READY
    client.send_audio(pcm_bytes)                            # repeat as audio arrives
    for ev in client.drain():                              # non-blocking-ish poll
        if ev.kind == "final": inject(ev.text)
    client.flush()                                         # utterance boundary
    client.bye()                                           # end session
"""

from __future__ import annotations

import socket
import sys

import constants
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Import the dependency-free wire module. Works whether run from the repo root
# or as a package.
_REPO_DIR = Path(__file__).resolve().parent.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from mumble_stt import protocol as proto  # noqa: E402


@dataclass
class StreamResult:
    """One sidecar->daemon result. ``kind`` is 'partial', 'final', or 'error'."""
    kind: str
    text: str = ""
    utt: int = 0
    code: str = ""        # set for kind == 'error'
    fatal: bool = False   # set for kind == 'error'


class SidecarUnavailable(RuntimeError):
    """The sidecar could not be reached, rejected the session, or dropped."""


class NemotronStreamingClient:
    """Blocking client for one streaming session against the sidecar."""

    def __init__(self, socket_path: str, connect_timeout: float = constants.SIDECAR_CONNECT_TIMEOUT_SECONDS):
        self.socket_path = socket_path
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None

    # --- session setup ------------------------------------------------------
    def connect(self, session_id: Optional[str] = None, sample_rate: int = 16000,
                want_partials: bool = True) -> dict:
        """Connect, send HELLO, and block until READY. Returns the READY dict.

        Raises SidecarUnavailable on connect failure, an ERROR{busy/bad_audio}
        reply, or an unexpected frame.
        """
        session_id = session_id or uuid.uuid4().hex
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        try:
            s.connect(self.socket_path)
        except OSError as e:
            s.close()
            raise SidecarUnavailable(f"connect to {self.socket_path} failed: {e}") from e

        self._sock = s
        self._send(proto.hello(session_id, sample_rate=sample_rate,
                               want_partials=want_partials))

        type_tag, payload = proto.read_frame(s)
        if type_tag == proto.TYPE_ERROR:
            info = proto.decode_json(payload)
            self.close()
            raise SidecarUnavailable(
                f"sidecar refused session: {info.get('code')}: {info.get('message')}")
        if type_tag != proto.TYPE_READY:
            self.close()
            raise SidecarUnavailable(
                f"expected READY, got {proto.TYPE_NAMES.get(type_tag, type_tag)}")
        s.settimeout(None)
        return proto.decode_json(payload)

    # --- streaming ----------------------------------------------------------
    def send_audio(self, pcm: bytes) -> None:
        """Send one AUDIO frame (raw s16le mono 16k PCM)."""
        self._send(proto.audio(pcm))

    def flush(self) -> None:
        """Signal an utterance boundary (finalize + reset, stay connected)."""
        self._send(proto.flush(0))

    def bye(self) -> None:
        """End the session cleanly; the sidecar keeps the model resident."""
        if self._sock is None:
            return
        try:
            self._send(proto.bye(uuid.uuid4().hex))
        except SidecarUnavailable:
            pass

    def read_result(self) -> Optional[StreamResult]:
        """Block for one sidecar->daemon frame. Returns None on clean EOF."""
        if self._sock is None:
            return None
        try:
            type_tag, payload = proto.read_frame(self._sock)
        except proto.ProtocolError:
            return None
        return self._to_result(type_tag, payload)

    def drain(self, timeout: float = 0.0) -> Iterator[StreamResult]:
        """Yield any results currently readable within ``timeout`` seconds.

        A convenience for poll-style use from a capture loop; for a dedicated
        reader thread, loop on :meth:`read_result` instead.
        """
        if self._sock is None:
            return
        # timeout=0.0 puts the socket in non-blocking mode, where a read with
        # no data ready raises BlockingIOError; a positive timeout raises
        # socket.timeout. Treat both as "nothing more right now". ProtocolError
        # means the peer closed mid-frame (clean EOF mid-read) -> stop.
        self._sock.settimeout(timeout)
        try:
            while True:
                try:
                    type_tag, payload = proto.read_frame(self._sock)
                except (socket.timeout, BlockingIOError):
                    return
                except proto.ProtocolError:
                    return
                res = self._to_result(type_tag, payload)
                if res is not None:
                    yield res
        finally:
            if self._sock is not None:
                self._sock.settimeout(None)

    @staticmethod
    def _to_result(type_tag: int, payload: bytes) -> Optional[StreamResult]:
        if type_tag == proto.TYPE_FINAL:
            obj = proto.decode_json(payload)
            return StreamResult(kind="final", text=obj.get("text", ""),
                                utt=int(obj.get("utt", 0)))
        if type_tag == proto.TYPE_PARTIAL:
            obj = proto.decode_json(payload)
            return StreamResult(kind="partial", text=obj.get("text", ""),
                                utt=int(obj.get("utt", 0)))
        if type_tag == proto.TYPE_ERROR:
            obj = proto.decode_json(payload)
            return StreamResult(kind="error", code=obj.get("code", ""),
                                text=obj.get("message", ""),
                                fatal=bool(obj.get("fatal", False)))
        return None  # READY/HELLO mid-session, ignore

    # --- io -----------------------------------------------------------------
    def _send(self, frame: bytes) -> None:
        if self._sock is None:
            raise SidecarUnavailable("not connected")
        try:
            self._sock.sendall(frame)
        except OSError as e:
            raise SidecarUnavailable(f"send failed: {e}") from e

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "NemotronStreamingClient":
        return self

    def __exit__(self, *exc) -> None:
        self.bye()
        self.close()


def _cli() -> int:
    """Stream a 16k mono s16le WAV through a running sidecar and print results.

    Usage: python -m backends.nemotron_streaming <socket_path> <file.wav>
    """
    import wave

    if len(sys.argv) != 3:
        sys.stderr.write("usage: nemotron_streaming.py <socket_path> <file.wav>\n")
        return 2
    sock_path, wav_path = sys.argv[1], sys.argv[2]

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())

    client = NemotronStreamingClient(sock_path)
    ready = client.connect(want_partials=True)
    print(f"READY: {ready}")
    # Feed ~100ms (3200-byte) chunks like the daemon does.
    step = 3200
    for i in range(0, len(pcm), step):
        client.send_audio(pcm[i:i + step])
        # Small drain window so partials surface as we go without racing the
        # ~9ms per-chunk inference.
        for res in client.drain(timeout=0.05):
            tag = res.kind.upper()
            print(f"  {tag}: {res.text or res.code}")
    client.flush()
    # Drain the final after flush.
    for res in client.drain(timeout=2.0):
        print(f"  {res.kind.upper()}: {res.text or res.code}")
    client.bye()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

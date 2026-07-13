import unittest

from stream_vad import StreamingVadGate, VadResult


def chunk(marker: int, milliseconds: int = 100) -> bytes:
    # 16 kHz mono s16le
    return bytes([marker]) * (16 * 2 * milliseconds)


class SequenceDetector:
    def __init__(self, results):
        self.results = iter(results)

    def process_pcm16(self, pcm):
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class TestStreamingVadGate(unittest.TestCase):
    def test_withholds_idle_then_flushes_bounded_pre_roll(self):
        detector = SequenceDetector(
            [
                VadResult(False),
                VadResult(False),
                VadResult(False),
                VadResult(True, speech_started=True),
            ]
        )
        gate = StreamingVadGate(detector, pre_roll_ms=200, trailing_ms=300)
        first, second, third, speech = [chunk(i) for i in range(1, 5)]

        self.assertEqual(gate.feed(first), [])
        self.assertEqual(gate.feed(second), [])
        self.assertEqual(gate.feed(third), [])
        self.assertEqual(gate.feed(speech), [second, third, speech])

    def test_ships_trailing_silence_then_resumes_gating(self):
        detector = SequenceDetector(
            [VadResult(True), VadResult(False, speech_ended=True), VadResult(False), VadResult(False)]
        )
        gate = StreamingVadGate(detector, pre_roll_ms=100, trailing_ms=200)
        frames = [chunk(i) for i in range(4)]

        self.assertEqual(gate.feed(frames[0]), [frames[0]])
        self.assertEqual(gate.feed(frames[1]), [frames[1]])
        self.assertEqual(gate.feed(frames[2]), [frames[2]])
        self.assertEqual(gate.feed(frames[3]), [])

    def test_detector_failure_flushes_pre_roll_and_fails_open(self):
        detector = SequenceDetector(
            [VadResult(False), RuntimeError("broken")]
        )
        gate = StreamingVadGate(detector, pre_roll_ms=200, trailing_ms=200)
        held, failed, later = chunk(1), chunk(2), chunk(3)

        self.assertEqual(gate.feed(held), [])
        self.assertEqual(gate.feed(failed), [held, failed])
        self.assertTrue(gate.failed)
        self.assertEqual(gate.feed(later), [later])

    def test_zero_pre_roll_drops_idle_without_buffering(self):
        gate = StreamingVadGate(
            SequenceDetector([VadResult(False)]), pre_roll_ms=0, trailing_ms=0
        )
        self.assertEqual(gate.feed(chunk(1)), [])


if __name__ == "__main__":
    unittest.main()

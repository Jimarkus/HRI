"""Mister Bookshelf voice front end.

    python main.py                     # listen forever, print each command
    python main.py --once              # return after one command
    python main.py --model small.en    # more accurate transcription (slower)
    python main.py --wake whisper      # force the no-training wake word backend

Pipeline: wake word "Mister Bookshelf" -> record until the speaker stops -> faster-whisper -> str
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from audio_io import CHUNK_SECONDS, SAMPLE_RATE, MicStream, SpeechDetector, int16_to_float
from fast_transcribe import Transcriber
from wakeword import build_detector


def beep(freq: int = 880, ms: int = 120) -> None:
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:  # noqa: BLE001 - non-Windows or no audio device: just skip the cue
        pass


def record_utterance(stream: MicStream, vad: SpeechDetector, start_timeout: float = 5.0,
                     end_silence: float = 0.8, max_duration: float = 15.0) -> np.ndarray | None:
    """Record from the moment speech starts until `end_silence` seconds of quiet.

    Returns int16 audio, or None if nobody spoke within `start_timeout` seconds.
    """
    vad.reset()
    preroll: deque[np.ndarray] = deque(maxlen=4)  # ~0.3 s so the first syllable isn't clipped
    chunks: list[np.ndarray] = []
    silence = 0
    waited = 0
    end_chunks = round(end_silence / CHUNK_SECONDS)
    max_chunks = round(max_duration / CHUNK_SECONDS)
    while True:
        chunk = stream.read()
        speaking = vad.is_speech(chunk)
        if not chunks:
            if speaking:
                chunks = list(preroll) + [chunk]
            else:
                preroll.append(chunk)
                waited += 1
                if waited * CHUNK_SECONDS >= start_timeout:
                    return None
            continue
        chunks.append(chunk)
        silence = 0 if speaking else silence + 1
        if silence >= end_chunks or len(chunks) >= max_chunks:
            return np.concatenate(chunks)


def save_wav(audio: np.ndarray, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / f"command_{datetime.now():%Y%m%d_%H%M%S}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())
    return path


class MisterBookshelf:
    """Owns the mic, wake word detector and transcriber. `listen()` returns one command string."""

    def __init__(self, model: str | None = None, wake: str = "auto", threshold: float = 0.5,
                 device: str = "auto", mic: int | None = None, save_dir: Path | None = None,
                 sound: bool = True):
        print("Loading models...")
        self.vad = SpeechDetector()
        self.detector = build_detector(self.vad, wake, threshold, device)
        self.transcriber = self._load_transcriber(model, device)
        self.stream = MicStream(mic)
        self.save_dir = save_dir
        self.sound = sound
        print(f"  mic:        {self.stream.device_name}")
        print(f"  wake word:  {self.detector.name}")
        print(f"  transcribe: faster-whisper {self.transcriber.model_name} on {self.transcriber.device}")

    @staticmethod
    def _load_transcriber(model: str | None, device: str) -> Transcriber:
        """No model given: small.en on the GPU (~0.3 s per command), base.en on CPU (~0.75 s)."""
        if model:
            return Transcriber(model, device)
        if device != "cpu":
            t = Transcriber("small.en", device)
            if t.device == "cuda":
                return t
        return Transcriber("base.en", "cpu")

    def listen(self) -> str:
        """Block until "Mister Bookshelf" + a command is heard; return the command text."""
        while True:
            self.stream.drain()
            print("\nListening for 'Mister Bookshelf'...")
            event = self.detector.wait(self.stream)
            print(f"Wake word detected (score {event.score:.2f}). Go ahead...")
            if self.sound:
                beep()
            self.stream.drain()  # drop audio captured during the beep

            # If the command was already said in the same breath, don't wait long for more.
            timeout = 1.5 if event.trailing_text else 5.0
            audio = record_utterance(self.stream, self.vad, start_timeout=timeout)
            parts = [event.trailing_text]
            if audio is not None:
                if self.save_dir:
                    print(f"  saved {save_wav(audio, self.save_dir)}")
                t0 = time.perf_counter()
                parts.append(self.transcriber.transcribe(int16_to_float(audio)))
                print(f"  transcribed {len(audio) / SAMPLE_RATE:.1f}s of audio "
                      f"in {time.perf_counter() - t0:.2f}s")
            text = " ".join(p for p in parts if p).strip()
            if text:
                if self.sound:
                    beep(660, 80)
                return text
            print("No command heard.")

    def close(self) -> None:
        self.stream.close()


def handle_command(text: str) -> None:
    """Hook for the rest of the robot: replace this with whatever acts on the command."""
    print(f'>>> "{text}"')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="faster-whisper model for commands: tiny.en, base.en, small.en, "
                                    "medium.en, turbo (default: small.en on GPU, base.en on CPU)")
    ap.add_argument("--wake", default="auto", choices=["auto", "oww", "whisper"],
                    help="auto = trained mister_bookshelf.onnx if present, else Whisper keyword spotting")
    ap.add_argument("--threshold", type=float, default=0.5, help="openWakeWord detection threshold")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--mic", type=int, help="PyAudio input device index (default: system default)")
    ap.add_argument("--save", action="store_true", help="save each command as a WAV in recordings/")
    ap.add_argument("--no-sound", action="store_true", help="disable the beeps")
    ap.add_argument("--once", action="store_true", help="exit after one command")
    args = ap.parse_args()

    bot = MisterBookshelf(args.model, args.wake, args.threshold, args.device, args.mic,
                          Path(__file__).parent / "recordings" if args.save else None,
                          sound=not args.no_sound)
    try:
        while True:
            handle_command(bot.listen())
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        bot.close()


if __name__ == "__main__":
    sys.exit(main())

"""Wake word detection for "Mister Bookshelf".

Two interchangeable backends, both exposing `wait(stream) -> WakeEvent`:

* OpenWakeWordDetector - uses the custom model produced by train_wakeword.py
  (mister_bookshelf.onnx). Tiny, always-on, very low CPU. Preferred.
* WhisperWakeDetector - fallback that works with no training: Silero VAD cuts the audio
  into utterances and a tiny Whisper model checks each one for the phrase. Higher CPU
  (only while someone is talking), but it bonus-captures a command spoken in the same
  breath ("Mister Bookshelf, find me Dune").
"""
from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_io import CHUNK_SECONDS, MicStream, SpeechDetector, int16_to_float

WAKE_PHRASE = "mister bookshelf"
MODEL_SEARCH_PATHS = [
    Path(__file__).parent / "mister_bookshelf.onnx",
    Path(__file__).parent / "models" / "mister_bookshelf.onnx",
    Path(__file__).parent / "oww_training" / "output" / "mister_bookshelf.onnx",
]


@dataclass
class WakeEvent:
    score: float
    trailing_text: str = ""  # words spoken right after the wake phrase (Whisper backend only)


def find_wakeword_model() -> Path | None:
    return next((p for p in MODEL_SEARCH_PATHS if p.is_file()), None)


class OpenWakeWordDetector:
    name = "openWakeWord"

    def __init__(self, model_path: Path, threshold: float = 0.5):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self.threshold = threshold

    def wait(self, stream: MicStream) -> WakeEvent:
        self.model.reset()
        while True:
            scores = self.model.predict(stream.read())
            score = max(scores.values())
            if score >= self.threshold:
                self.model.reset()
                return WakeEvent(score=score)


_ALIASES = {"mr": "mister", "misterbookshelf": "mister bookshelf"}


def _normalize(text: str) -> list[str]:
    text = re.sub(r"[^a-z' ]+", " ", text.lower().replace("-", " "))
    words = [_ALIASES.get(w, w) for w in text.split()]
    return " ".join(words).split()


def match_wake_phrase(text: str, min_ratio: float = 0.84) -> tuple[float, str] | None:
    """Fuzzy-find the wake phrase in a transcript.

    Whisper writes it many ways ("Mr. Bookshelf", "Mister book shelf", "Mr Bookself"...), so we
    compare every 1-3 word window, with spaces removed, against "misterbookshelf".
    Returns (similarity, text after the phrase) or None.
    """
    words = _normalize(text)
    target = WAKE_PHRASE.replace(" ", "")
    best = None
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            candidate = "".join(words[i:i + n])
            ratio = difflib.SequenceMatcher(None, candidate, target).ratio()
            if ratio >= min_ratio and (best is None or ratio > best[0]):
                best = (ratio, i + n)
    if best is None:
        return None
    # recover the trailing words from the original (un-normalized) transcript
    original = text.split()
    if len(original) == len(words):
        trailing = " ".join(original[best[1]:]).lstrip(" ,.!?")
    else:  # word boundaries shifted during normalization; fall back to the normalized words
        trailing = " ".join(words[best[1]:])
    return best[0], trailing


class WhisperWakeDetector:
    name = "Whisper keyword spotting"

    def __init__(self, vad: SpeechDetector, model_name: str = "tiny.en", device: str = "auto",
                 end_silence: float = 0.4, max_segment: float = 6.0):
        from fast_transcribe import Transcriber
        self.transcriber = Transcriber(model_name, device, hotwords="Mister Bookshelf")
        self.vad = vad
        self.end_silence_chunks = max(1, round(end_silence / CHUNK_SECONDS))
        self.max_segment_chunks = round(max_segment / CHUNK_SECONDS)
        self.preroll = 3  # chunks of audio kept from before speech started

    def wait(self, stream: MicStream) -> WakeEvent:
        self.vad.reset()
        history: deque[np.ndarray] = deque(maxlen=self.preroll)
        segment: list[np.ndarray] = []
        silence = 0
        while True:
            chunk = stream.read()
            speaking = self.vad.is_speech(chunk)
            if not segment:
                if speaking:
                    segment = list(history) + [chunk]
                    silence = 0
                else:
                    history.append(chunk)
                continue
            segment.append(chunk)
            silence = 0 if speaking else silence + 1
            if silence >= self.end_silence_chunks or len(segment) >= self.max_segment_chunks:
                audio = int16_to_float(np.concatenate(segment))
                segment, history = [], deque(maxlen=self.preroll)
                text = self.transcriber.transcribe(audio, beam_size=1)
                hit = match_wake_phrase(text)
                if hit:
                    return WakeEvent(score=hit[0], trailing_text=hit[1])
                stream.drain()  # skip audio that queued up during transcription


def build_detector(vad: SpeechDetector, backend: str = "auto", threshold: float = 0.5,
                   device: str = "auto"):
    model_path = find_wakeword_model()
    if backend == "oww" or (backend == "auto" and model_path):
        if not model_path:
            raise FileNotFoundError("No mister_bookshelf.onnx found. Train one with "
                                    "`python train_wakeword.py \"mister bookshelf\"` or use --wake whisper.")
        return OpenWakeWordDetector(model_path, threshold)
    return WhisperWakeDetector(vad, device=device)

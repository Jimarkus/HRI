"""Microphone input and voice-activity detection shared by every stage of the pipeline.

One microphone stream is opened for the whole session and read in fixed 80 ms chunks
(1280 samples at 16 kHz), which is exactly the frame size openWakeWord expects.
"""
from __future__ import annotations

import numpy as np
import pyaudio

SAMPLE_RATE = 16_000
CHUNK = 1280  # 80 ms at 16 kHz
CHUNK_SECONDS = CHUNK / SAMPLE_RATE


class MicStream:
    """Mono 16 kHz int16 microphone stream that yields CHUNK-sized numpy arrays.

    Asks the driver for 16 kHz directly; if the device refuses, it opens at the device's
    native rate and resamples in software.
    """

    def __init__(self, device_index: int | None = None):
        self._pa = pyaudio.PyAudio()
        info = (self._pa.get_device_info_by_index(device_index) if device_index is not None
                else self._pa.get_default_input_device_info())
        self.device_name = info["name"]
        self._native_rate = SAMPLE_RATE
        try:
            self._stream = self._open(device_index, SAMPLE_RATE, CHUNK)
        except OSError:
            self._native_rate = int(info["defaultSampleRate"])
            frames = int(round(CHUNK * self._native_rate / SAMPLE_RATE))
            self._stream = self._open(device_index, self._native_rate, frames)
        self._native_frames = int(round(CHUNK * self._native_rate / SAMPLE_RATE))

    def _open(self, device_index, rate, frames):
        return self._pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                             input_device_index=device_index, frames_per_buffer=frames)

    def read(self) -> np.ndarray:
        """Block until the next 80 ms chunk is available; returns int16 array of length CHUNK."""
        raw = self._stream.read(self._native_frames, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.int16)
        if self._native_rate != SAMPLE_RATE:
            x_old = np.linspace(0, 1, num=audio.size, endpoint=False)
            x_new = np.linspace(0, 1, num=CHUNK, endpoint=False)
            audio = np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.int16)
        return audio

    def drain(self) -> None:
        """Discard audio that piled up while we were busy (e.g. transcribing)."""
        available = self._stream.get_read_available()
        if available > 0:
            self._stream.read(available, exception_on_overflow=False)

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SpeechDetector:
    """Silero VAD (the copy bundled with openWakeWord). Returns P(speech) for an 80 ms chunk."""

    def __init__(self, threshold: float = 0.5):
        from openwakeword.vad import VAD
        self._vad = VAD()
        self.threshold = threshold

    def probability(self, chunk: np.ndarray) -> float:
        return float(self._vad.predict(chunk, frame_size=640))

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.probability(chunk) >= self.threshold

    def reset(self) -> None:
        self._vad.reset_states()


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768.0

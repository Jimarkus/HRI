"""Fast speech-to-text with faster-whisper (CTranslate2).

Used two ways:
  * as a library:  Transcriber("base.en").transcribe(float32_audio_16k) -> str
  * as a CLI:      python fast_transcribe.py recordings/test1.mp3 [--model small.en]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows cache warning


def _add_nvidia_dll_dirs() -> None:
    """Make the cuBLAS DLLs from the `nvidia-cublas-cu12` pip wheel findable on Windows.

    CTranslate2 loads cublas64_12.dll lazily via LoadLibrary, which searches PATH, so the
    wheel's bin folders are prepended to PATH (and registered with add_dll_directory).
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
            os.add_dll_directory(str(bin_dir))


_add_nvidia_dll_dirs()

from faster_whisper import WhisperModel


def load_audio_file(path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 using ffmpeg (via openai-whisper's loader)."""
    from whisper.audio import load_audio
    return load_audio(str(path))


class Transcriber:
    def __init__(self, model_name: str = "base.en", device: str = "auto", hotwords: str | None = None):
        self.model_name = model_name
        self.hotwords = hotwords
        self.model, self.device = self._load(model_name, device)
        self.language = "en" if model_name.endswith(".en") else None

    @staticmethod
    def _load(model_name: str, device: str):
        """Prefer the GPU, but fall back to CPU int8 if CUDA libraries (cuBLAS/cuDNN) are missing."""
        attempts = []
        if device in ("auto", "cuda"):
            attempts.append(("cuda", "float16"))
        if device in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))
        last_error = None
        for dev, compute_type in attempts:
            try:
                model = WhisperModel(model_name, device=dev, compute_type=compute_type)
                # CUDA DLL problems only surface on first use, so run a tiny warm-up.
                list(model.transcribe(np.zeros(16_000, dtype=np.float32), beam_size=1)[0])
                return model, dev
            except Exception as e:  # noqa: BLE001 - any failure means "try the next device"
                last_error = e
        raise RuntimeError(f"Could not load Whisper model {model_name!r}: {last_error}")

    def transcribe(self, audio, beam_size: int = 5) -> str:
        """audio: float32 numpy array at 16 kHz, or a path to an audio file."""
        if isinstance(audio, (str, Path)):
            audio = load_audio_file(audio)
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            condition_on_previous_text=False,
            without_timestamps=True,
            hotwords=self.hotwords,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    ap.add_argument("audio", nargs="?", default=str(Path(__file__).parent / "recordings" / "test1.mp3"))
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    t = Transcriber(args.model, args.device)
    print(f"Loaded {args.model} on {t.device} in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    text = t.transcribe(Path(args.audio))
    print(f"[{time.perf_counter() - t0:.2f}s] {text}")


if __name__ == "__main__":
    main()

# HRI
HRI Book pusher project repo

## Mister Bookshelf voice front end

Say **"Mister Bookshelf"**, then your command. The program records until you stop
talking, transcribes it with faster-whisper, and returns the text as a string.

```
python main.py                   # listen forever, print each command
python main.py --once            # stop after one command
python main.py --model small.en  # more accurate, ~3x slower
python main.py --save            # also keep each command as a WAV in recordings/
python fast_transcribe.py some_file.mp3   # transcribe a file
```

Setup: `pip install -r requirements.txt` (ffmpeg must be on PATH for file transcription).

### Pipeline
1. **Wake word** (`wakeword.py`)
   - If `mister_bookshelf.onnx` exists (made by `python train_wakeword.py "mister bookshelf"`,
     needs Linux/WSL2), openWakeWord runs on it: always-on and very cheap.
   - Otherwise it falls back to Whisper keyword spotting: Silero VAD finds utterances and
     `tiny.en` checks each one for the phrase. Works without training. It also catches
     "Mister Bookshelf, find me Dune" said in one breath.
2. **Record** (`main.py: record_utterance`): starts when speech starts, stops after
   0.8 s of silence (Silero VAD). Gives up if nobody speaks within 5 s.
3. **Transcribe** (`fast_transcribe.py`): faster-whisper. On the GPU (CUDA via the
   `nvidia-cublas-cu12` wheel) it uses `small.en` in float16, ~0.3 s for 8 s of audio. Without a
   GPU it falls back to `base.en` int8 on CPU, ~0.75 s.

### Using it from other code
```python
from main import MisterBookshelf
bot = MisterBookshelf()
text = bot.listen()   # blocks until "Mister Bookshelf <command>", returns the command
```
Or edit `handle_command()` in `main.py`.

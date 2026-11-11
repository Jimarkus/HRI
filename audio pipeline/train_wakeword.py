#!/usr/bin/env python3
"""
train_wakeword.py - train a custom openWakeWord wake word with one command.

    python train_wakeword.py "mister bookshelf"

What it does (everything lives inside --workdir, nothing touches your system Python):
  1. Builds its own isolated Python 3.11 environment with pinned, tested package versions
  2. Clones openWakeWord and the Piper sample generator at pinned commits
  3. Downloads the text-to-speech model, noise/reverb data and pre-computed negative features
  4. Generates synthetic speech of your wake word (plus confusable "adversarial" phrases)
  5. Augments it, trains the model with openWakeWord's own train.py, exports <name>.onnx
  6. Loads the result back through openWakeWord as a sanity check

Re-running the same command resumes where it stopped (every step skips finished work).

Requirements: Linux (or WSL2 on Windows), an internet connection, ~35 GB free disk,
an NVIDIA GPU strongly recommended (CPU works but speech generation is slow).
The only things needed on the host are python3 (3.8+) and either `uv` or the `venv` module.

Useful options:
    --preview        only synthesize a few clips so you can hear how the TTS says your word
    --preset quick   smaller/faster run (less accurate); default is "standard"
    --self-test      run the whole pipeline on tiny synthetic data first, to prove your machine works
                     (~5 min of compute on a 2-core CPU, ~8 GB of disk mostly PyTorch; skips the big data downloads)
    --tflite         also convert to .tflite (needed by some integrations, e.g. Home Assistant)
    --cpu            ignore the GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------------------
# Pinned versions. These were chosen because they were tested together; see notes below.
# --------------------------------------------------------------------------------------
OWW_REPO = "https://github.com/dscripka/openWakeWord"
OWW_COMMIT = "368c03716d1e92591906a84949bc477f3a834455"
# The fork that openWakeWord's own config file points to. It uses espeak-phonemizer, so it
# does not need `piper-phonemize` (which has no wheels for modern Python versions).
PSG_REPO = "https://github.com/dscripka/piper-sample-generator"
PSG_COMMIT = "f1988a4d54eddb23d99e86f0adfef6226a85acc7"
TTS_MODEL_URL = ("https://github.com/rhasspy/piper-sample-generator/releases/download/"
                 "v1.0.0/en-us-libritts-high.pt")
OWW_RESOURCE_URLS = [
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
]
# only needed to load .tflite models through openWakeWord (--tflite)
OWW_RESOURCE_URLS_TFLITE = [
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
]
HF_RIR_REPO = "davidscripka/MIT_environmental_impulse_responses"
HF_AUDIOSET_REPO = "agkphysics/AudioSet"
HF_FEATURES_REPO = "davidscripka/openwakeword_features"
ACAV_FILE = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FILE = "validation_set_features.npy"

PYTHON_VERSION = "3.11"
# Resolve transitive dependencies as they existed when this script was tested.
EXCLUDE_NEWER = "2026-10-21T00:00:00Z"
# Notes on pins:
#  - torch 2.5.1: newer torch (>=2.6) breaks the sample generator's torch.load of its model
#    and (>=2.9) changes torch.onnx.export; torchaudio must match torch exactly.
#  - numpy<2 and scipy<1.15: `acoustics` needs scipy.special.sph_harm (removed in scipy 1.17).
#  - setuptools<81: `pronouncing` imports pkg_resources.
#  - huggingface_hub 0.36.x: a stable, well known API for the download helpers below.
REQUIREMENTS = """\
torch==2.5.1
torchaudio==2.5.1
numpy==1.26.4
scipy==1.14.1
soundfile==0.14.0
scikit-learn==1.9.1
requests==2.34.2
pyyaml==6.0.3
tqdm==4.70.1
onnxruntime==1.31.0
onnx==1.23.2
espeak-phonemizer==1.3.1
espeakng-loader==0.2.4
webrtcvad-wheels==2.0.14.post1
huggingface_hub==0.36.2
setuptools==80.10.2
audiomentations==0.33.0
torch-audiomentations==0.11.0
acoustics==0.2.6
speechbrain==0.5.14
torchinfo==1.8.0
torchmetrics==1.2.0
mutagen==1.47.0
pronouncing==0.2.0
"""
# Only used for --tflite (kept in a separate environment so it can never break training).
TFLITE_REQUIREMENTS = """\
tensorflow-cpu
ai-edge-litert
onnx2tf
onnx
onnxruntime
onnx_graphsurgeon
sng4onnx
onnxsim
psutil
protobuf
"""

PRESETS = {
    # n_samples, n_samples_val, steps
    "quick": (3000, 1000, 10000),
    "standard": (10000, 2000, 50000),
}


# --------------------------------------------------------------------------------------
# Small helpers (host side, standard library only)
# --------------------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def banner(msg: str) -> None:
    say("\n" + "=" * 78)
    say(msg)
    say("=" * 78)


def die(msg: str, code: int = 1) -> None:
    say(f"\nERROR: {msg}")
    sys.exit(code)


def sanitize_name(phrase: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return name or "my_wakeword"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def run(cmd, *, cwd=None, env=None, check=True, quiet=False):
    """Run a command, streaming its output. Returns the CompletedProcess."""
    if not quiet:
        say("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if check and proc.returncode != 0:
        die(f"command failed with exit code {proc.returncode}: {' '.join(str(c) for c in cmd)}")
    return proc


def run_logged(cmd, *, cwd, env, log_path: Path, title: str) -> None:
    """Run a long command: stream to the console, append to a log file, show the tail on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    say("$ " + " ".join(str(c) for c in cmd))
    say(f"  (full log: {log_path})")
    start = time.time()
    tail = b""
    with open(log_path, "ab") as log:
        log.write(f"\n\n##### {title} @ {time.ctime()} #####\n".encode())
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            tail = (tail + chunk)[-6000:]
        rc = proc.wait()
    mins = (time.time() - start) / 60
    if rc != 0:
        text = tail.decode("utf-8", "replace").replace("\r", "\n")
        say("\n--- last output ---\n" + "\n".join(text.splitlines()[-40:]) + "\n-------------------")
        die(f"'{title}' failed after {mins:.1f} min (exit code {rc}).\n"
            f"       Full log: {log_path}\n"
            "       Fix the problem (or just re-run the same command, finished steps are skipped).")
    say(f"\n[{title}] finished in {mins:.1f} min")


def download(url: str, dest: Path, what: str) -> None:
    """Download with a .part file so an interrupted download is never mistaken for a finished one."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    say(f"Downloading {what} ...")
    last = [0.0]

    def hook(blocks, block_size, total):
        now = time.time()
        if now - last[0] > 3 and total > 0:
            last[0] = now
            say(f"   {min(blocks * block_size, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB")

    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, part, hook)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                die(f"could not download {what} from {url}: {exc}")
            say(f"   retrying after error: {exc}")
            time.sleep(3)
    part.rename(dest)


def git_checkout(repo_url: str, commit: str, dest: Path) -> None:
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
    if not shutil.which("git"):
        die("git is required but was not found. Install it (e.g. `sudo apt install git`) and re-run.")
    if dest.exists():
        shutil.rmtree(dest)
    run(["git", "clone", "-q", repo_url, dest])
    run(["git", "-C", dest, "checkout", "-q", commit])


def confirm(question: str, assume_yes: bool) -> None:
    if assume_yes or not sys.stdin.isatty():
        return
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        die("aborted.", 0)


# --------------------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------------------
def find_uv(workdir: Path) -> str:
    found = shutil.which("uv")
    if found:
        return found
    boot = workdir / ".uv-bootstrap"
    uv = boot / "bin" / "uv"
    if uv.exists():
        return str(uv)
    say("`uv` not found; installing it into a private folder (needs the python3 `venv` module) ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(boot)], check=True)
        subprocess.run([str(boot / "bin" / "pip"), "install", "-q", "uv"], check=True)
        return str(uv)
    except Exception as exc:  # noqa: BLE001
        die("could not install `uv` automatically "
            f"({exc}).\n       Install it yourself with:\n"
            "           curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "       (or `sudo apt install python3-venv`) and re-run this script.")
    return ""


def ensure_env(uv: str, venv_dir: Path, requirements: str, extra_marker: str, editable: Path | None) -> None:
    tag = hashlib.sha1((requirements + extra_marker + PYTHON_VERSION + EXCLUDE_NEWER).encode()).hexdigest()[:12]
    marker = venv_dir / f".env-ok-{tag}"
    if marker.exists():
        return
    banner(f"Building isolated Python {PYTHON_VERSION} environment ({venv_dir.name}) - first run only")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    # --clear: never stop to ask "replace existing environment?"
    run([uv, "venv", "--clear", "--python", PYTHON_VERSION, venv_dir])
    req_file = venv_dir.parent / f"requirements-{venv_dir.name}.txt"
    req_file.write_text(requirements)
    py = venv_python(venv_dir)
    run([uv, "pip", "install", "--python", py, "-r", req_file, "--exclude-newer", EXCLUDE_NEWER])
    if editable is not None:
        run([uv, "pip", "install", "--python", py, "--no-deps", "-e", editable,
             "--exclude-newer", EXCLUDE_NEWER])
    marker.write_text("ok\n")


def patch_piper_torch_load(psg_dir: Path) -> None:
    """Newer torch versions default torch.load(weights_only=True), which rejects this model file."""
    path = psg_dir / "generate_samples.py"
    text = path.read_text()
    if "torch.load(model_path)" in text:
        path.write_text(text.replace("torch.load(model_path)", "torch.load(model_path, weights_only=False)"))


def patch_train_tflite_flag(oww_dir: Path) -> None:
    """Upstream bug: `--convert_to_tflite` defaults to the *string* "False", which is truthy, so train.py
    always tries a TensorFlow conversion (and crashes without the ancient onnx_tf) right after saving the
    ONNX model. Only convert when the flag was really passed."""
    path = oww_dir / "openwakeword" / "train.py"
    text = path.read_text()
    old = "if args.convert_to_tflite:"
    if old in text:
        path.write_text(text.replace(old, "if args.convert_to_tflite is True:"))
    elif "if args.convert_to_tflite is True:" not in text:
        say("WARNING: could not apply the tflite-flag patch to train.py; if training ends with an "
            "'onnx_tf' error, the .onnx file in the output folder is still valid.")


def espeak_env(venv_dir: Path, base: dict) -> dict:
    """Point the phonemizer at the espeak-ng library bundled in the pip package (no sudo/apt needed)."""
    out = subprocess.run([str(venv_python(venv_dir)), "-c",
                          "import espeakng_loader,os;print(os.path.dirname(espeakng_loader.get_library_path()))"],
                         capture_output=True, text=True)
    loader_dir = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else ""
    if not loader_dir:
        die("the bundled espeak-ng library could not be located:\n" + out.stderr[-1500:])
    env = dict(base)
    env["LD_LIBRARY_PATH"] = loader_dir + (":" + base["LD_LIBRARY_PATH"] if base.get("LD_LIBRARY_PATH") else "")
    env["ESPEAK_DATA_PATH"] = loader_dir
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def worker(py: Path, env: dict, stage: str, payload: dict, capture: bool = False):
    """Run one of this file's own worker stages inside the isolated environment."""
    cmd = [py, os.path.abspath(__file__), "_worker", stage, json.dumps(payload)]
    if capture:
        proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
        return proc
    run(cmd, env=env)
    return None


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train a custom openWakeWord wake word.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("phrase", nargs="?", default=None,
                    help='wake word/phrase, e.g. "mister bookshelf" (use spaces, not underscores)')
    ap.add_argument("--workdir", default="./oww_training", help="where everything is stored (default: ./oww_training)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    ap.add_argument("--n-samples", type=int, help="override: number of training clips to synthesize")
    ap.add_argument("--n-samples-val", type=int, help="override: number of validation clips")
    ap.add_argument("--steps", type=int, help="override: training steps")
    ap.add_argument("--negative-phrase", action="append", default=[],
                    help="a phrase the model must NOT trigger on (repeatable)")
    ap.add_argument("--audioset-shards", default="bal_train09",
                    help="comma separated AudioSet shards used as background noise/music (default: bal_train09)")
    ap.add_argument("--preview", action="store_true", help="only synthesize a few clips to hear the pronunciation")
    ap.add_argument("--tflite", action="store_true", help="also convert the model to .tflite")
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    ap.add_argument("--self-test", action="store_true", help="run the full pipeline on tiny synthetic data")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before big downloads")
    ap.add_argument("--keep-downloads", action="store_true", help="keep downloaded archives after conversion")
    args = ap.parse_args()

    if platform.system() != "Linux":
        die("this pipeline needs Linux (the speech-synthesis library is Linux-only).\n"
            "       On Windows use WSL2 (Ubuntu) and run the script there; on macOS use a Linux machine,\n"
            "       a VM, or Google Colab (`!python train_wakeword.py ...`).")
    if platform.machine() not in ("x86_64", "AMD64", "aarch64"):
        die(f"unsupported CPU architecture: {platform.machine()}")

    phrase = (args.phrase or ("hey tester" if args.self_test else "")).replace("_", " ").strip()
    if not phrase:
        ap.error('please give the wake word, e.g.  python train_wakeword.py "mister bookshelf"')
    model_name = sanitize_name(phrase)

    n_samples, n_val, steps = PRESETS[args.preset]
    n_samples = args.n_samples or n_samples
    n_val = args.n_samples_val or n_val
    steps = args.steps or steps
    if args.self_test:
        n_samples = args.n_samples or 60
        n_val = args.n_samples_val or 20
        steps = args.steps or 200

    launch_dir = Path.cwd()
    workdir = Path(args.workdir).expanduser().resolve()
    if args.self_test:
        workdir = workdir / "selftest"
    workdir.mkdir(parents=True, exist_ok=True)
    logs = workdir / "logs"

    banner(f'openWakeWord training: "{phrase}"  (model name: {model_name})')
    say(f"working folder : {workdir}")
    say(f"clips/validation/steps : {n_samples} / {n_val} / {steps}" + ("   [self-test]" if args.self_test else ""))

    # ---- 1. environment -------------------------------------------------------------
    uv = find_uv(workdir)
    oww_dir = workdir / "openWakeWord"
    psg_dir = workdir / "piper-sample-generator"
    banner("Fetching code (pinned commits)")
    git_checkout(OWW_REPO, OWW_COMMIT, oww_dir)
    git_checkout(PSG_REPO, PSG_COMMIT, psg_dir)
    patch_piper_torch_load(psg_dir)
    patch_train_tflite_flag(oww_dir)

    venv_dir = workdir / ".venv"
    ensure_env(uv, venv_dir, REQUIREMENTS, OWW_COMMIT, oww_dir)
    py = venv_python(venv_dir)
    env = espeak_env(venv_dir, os.environ.copy())

    # ---- 2. models ------------------------------------------------------------------
    banner("Downloading models")
    download(TTS_MODEL_URL, psg_dir / "models" / "en-us-libritts-high.pt", "text-to-speech model (~250 MB)")
    res_dir = oww_dir / "openwakeword" / "resources" / "models"
    for url in OWW_RESOURCE_URLS + (OWW_RESOURCE_URLS_TFLITE if args.tflite else []):
        download(url, res_dir / url.rsplit("/", 1)[1], url.rsplit("/", 1)[1])

    # ---- 3. GPU check ---------------------------------------------------------------
    use_gpu = False
    if not args.cpu:
        proc = worker(py, env, "gpucheck", {}, capture=True)
        info = {}
        for line in (proc.stdout or "").splitlines():
            if line.startswith("{"):
                info = json.loads(line)
        if info.get("ok"):
            use_gpu = True
            say(f"GPU: {info.get('name')} (will be used for speech synthesis and training)")
        else:
            say(f"GPU not usable ({info.get('reason', 'none found')}) - running on CPU.")
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        say("NOTE: on CPU, speech generation takes a long time. Consider --preset quick or fewer --n-samples.")

    # ---- 4. preview mode ------------------------------------------------------------
    if args.preview:
        out = workdir / "preview" / model_name
        worker(py, env, "preview", {"psg": str(psg_dir), "phrase": phrase, "out": str(out), "n": 6})
        banner("Preview ready")
        say(f"Listen to the files in: {out}")
        say("If the word is mispronounced, respell it phonetically (e.g. 'kwee-noh' for 'quinoa') and re-run.")
        return

    # ---- 5. data --------------------------------------------------------------------
    banner("Preparing training data")
    if args.self_test:
        worker(py, env, "synth", {"workdir": str(workdir)})
    else:
        free_gb = shutil.disk_usage(workdir).free / 1e9
        say(f"free disk space here: {free_gb:,.0f} GB")
        if free_gb < 35 and not (workdir / "features" / ACAV_FILE).exists():
            confirm("Less than ~35 GB free. The negative-feature file alone is very large. Continue anyway?",
                    args.yes)
        confirm("About to download several GB of public training data from HuggingFace. Continue?", args.yes)
        worker(py, env, "fetch", {"workdir": str(workdir), "shards": args.audioset_shards,
                                  "keep": args.keep_downloads})

    # ---- 6. config ------------------------------------------------------------------
    cfg_path = workdir / f"{model_name}.yaml"
    worker(py, env, "config", {
        "workdir": str(workdir), "oww": str(oww_dir), "psg": str(psg_dir), "phrase": phrase,
        "model_name": model_name, "n_samples": n_samples, "n_val": n_val, "steps": steps,
        "negatives": args.negative_phrase, "out": str(cfg_path), "self_test": args.self_test,
    })

    # ---- 7. generate / augment / train (openWakeWord's own train.py, unmodified) ----
    train_py = oww_dir / "openwakeword" / "train.py"
    banner("Step 1/3: generating synthetic speech")
    run_logged([py, "-u", train_py, "--training_config", cfg_path, "--generate_clips"],
               cwd=workdir, env=env, log_path=logs / "1_generate.log", title="generate clips")

    banner("Step 2/3: augmenting clips and computing features")
    # Hide the GPU here on purpose: train.py would otherwise request onnxruntime's CUDA provider,
    # which the plain `onnxruntime` package does not have. CPU is reliable and fast enough.
    env_cpu = dict(env, CUDA_VISIBLE_DEVICES="")
    run_logged([py, "-u", train_py, "--training_config", cfg_path, "--augment_clips"],
               cwd=workdir, env=env_cpu, log_path=logs / "2_augment.log", title="augment clips")

    banner("Step 3/3: training the model")
    run_logged([py, "-u", train_py, "--training_config", cfg_path, "--train_model"],
               cwd=workdir, env=env, log_path=logs / "3_train.log", title="train model")

    onnx_src = workdir / "output" / f"{model_name}.onnx"
    if not onnx_src.exists():
        die(f"training finished but {onnx_src} was not created. See {logs / '3_train.log'}")

    # ---- 8. optional tflite -----------------------------------------------------------
    tflite_src = None
    if args.tflite:
        tflite_src = convert_tflite(uv, workdir, onnx_src, model_name, env, py)

    # ---- 9. verify through openWakeWord ------------------------------------------------
    banner("Checking the model through openWakeWord")
    worker(py, env_cpu, "verify", {"onnx": str(onnx_src), "clips_dir": str(workdir / "output" / model_name),
                                   "tflite": str(tflite_src) if tflite_src else ""})

    # ---- 10. deliver ----------------------------------------------------------------
    dest_onnx = launch_dir / f"{model_name}.onnx"
    shutil.copy2(onnx_src, dest_onnx)
    dest_tflite = None
    if tflite_src:
        dest_tflite = launch_dir / f"{model_name}.tflite"
        shutil.copy2(tflite_src, dest_tflite)

    banner("Done")
    say(f"Model file : {dest_onnx}")
    if dest_tflite:
        say(f"TFLite file: {dest_tflite}")
    say(f"""
Use it with openWakeWord (pip install openwakeword):

    import openwakeword
    from openwakeword.model import Model

    openwakeword.utils.download_models()          # one time: shared feature models
    oww = Model(wakeword_models=[r"{dest_onnx}"], inference_framework="onnx")
    scores = oww.predict(audio_frame)              # 16 kHz int16 numpy audio, e.g. 1280 samples
    print(scores["{model_name}"])                  # > 0.5 means "wake word heard"

Or try it live with the example script from the openWakeWord repo:

    python examples/detect_from_microphone.py --model_path "{dest_onnx}" --inference_framework onnx

NOTE: openWakeWord defaults to inference_framework="tflite". For an .onnx file you MUST pass "onnx".
Tune the 0.5 threshold for your room/microphone: raise it for fewer false triggers.
""")


# --------------------------------------------------------------------------------------
# TFLite conversion (separate environment; optional)
# --------------------------------------------------------------------------------------
def convert_tflite(uv: str, workdir: Path, onnx_src: Path, model_name: str, base_env: dict, main_py: Path):
    banner("Converting to TFLite (separate environment)")
    venv_dir = workdir / ".venv-tflite"
    try:
        ensure_env(uv, venv_dir, TFLITE_REQUIREMENTS, "tflite-v1", None)
        py = venv_python(venv_dir)
        out_dir = workdir / "output" / "tflite_build"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        env = dict(base_env, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="2")
        # onnx2tf shells out to helper programs (e.g. onnxsim) that live next to this environment's python
        env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
        run([py, os.path.abspath(__file__), "_worker", "tflite", json.dumps(
            {"onnx": str(onnx_src), "out": str(out_dir), "name": model_name})], env=env)
        produced = out_dir / f"{model_name}.tflite"
        if produced.exists():
            # openWakeWord needs the LiteRT runtime to load .tflite files (used for the check below)
            run([uv, "pip", "install", "--python", main_py, "ai-edge-litert==2.3.0", "numpy==1.26.4",
                 "--exclude-newer", EXCLUDE_NEWER])
            return produced
        say("WARNING: tflite conversion produced no file; the .onnx model is still fine.")
    except SystemExit:
        say("WARNING: tflite conversion failed; the .onnx model is still fine and usable.")
    return None


# ======================================================================================
# Worker stages - these run INSIDE the isolated environment (heavy imports live here)
# ======================================================================================
def _to_16k_int16(path):
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g)
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def _write_wav(path, data):
    import scipy.io.wavfile
    scipy.io.wavfile.write(str(path), 16000, data)


def _retry(fn, what, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"   {what}: retrying after error: {exc}", flush=True)
            time.sleep(5)


def _safe_extract(tar_path, dest):
    import tarfile
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not (m.isfile() or m.isdir()):
                continue
            if target != dest and not str(target).startswith(str(dest) + os.sep):
                continue
            tf.extract(m, dest)


def w_gpucheck(_: dict) -> None:
    info = {"ok": False, "reason": "no CUDA device"}
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            try:
                x = torch.ones(64, 64, device="cuda")
                float((x @ x).sum().item())
                info = {"ok": True, "name": name}
            except Exception as exc:  # noqa: BLE001
                info = {"ok": False, "reason": f"{name} cannot run this PyTorch build ({str(exc)[:120]})"}
    except Exception as exc:  # noqa: BLE001
        info = {"ok": False, "reason": str(exc)[:160]}
    print(json.dumps(info))


def w_preview(p: dict) -> None:
    sys.path.insert(0, p["psg"])
    os.chdir(p["psg"])
    from generate_samples import generate_samples
    Path(p["out"]).mkdir(parents=True, exist_ok=True)
    generate_samples(text=[p["phrase"]], max_samples=p["n"], batch_size=min(p["n"], 6),
                     output_dir=p["out"], length_scales=[1.0], auto_reduce_batch_size=True)


def w_fetch(p: dict) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from tqdm import tqdm

    wd = Path(p["workdir"])
    dl = wd / "downloads"
    dl.mkdir(parents=True, exist_ok=True)

    # --- room impulse responses (small) ---
    rir_out = wd / "mit_rirs"
    if not rir_out.exists() or not any(rir_out.glob("*.wav")):
        print("\n[data] room impulse responses ...", flush=True)
        local = _retry(lambda: snapshot_download(HF_RIR_REPO, repo_type="dataset",
                                                 allow_patterns=["16khz/*.wav"], local_dir=str(dl / "mit_rirs_raw")),
                       "RIR download")
        files = sorted(Path(local).glob("16khz/*.wav"))
        if not files:
            raise SystemExit("no RIR wav files were found in the downloaded dataset")
        rir_out.mkdir(exist_ok=True)
        for f in tqdm(files, desc="RIRs"):
            _write_wav(rir_out / f.name, _to_16k_int16(f))

    # --- background noise / music from AudioSet ---
    bg_out = wd / "audioset_16k"   # must contain ONLY audio files: train.py treats every file in it as audio
    bg_out.mkdir(exist_ok=True)
    (dl / "flags").mkdir(exist_ok=True)
    for shard in [s.strip() for s in p["shards"].split(",") if s.strip()]:
        done_flag = dl / "flags" / f"{shard}.done"
        if done_flag.exists():
            continue
        print(f"\n[data] AudioSet shard {shard} (background audio) ...", flush=True)
        tar_path = _retry(lambda: hf_hub_download(HF_AUDIOSET_REPO, f"data/{shard}.tar", repo_type="dataset",
                                                  local_dir=str(dl / "audioset")), "AudioSet download")
        ex_dir = dl / "audioset_extract" / shard
        _safe_extract(tar_path, ex_dir)
        flacs = sorted(Path(ex_dir).glob("**/*.flac"))
        if not flacs:
            raise SystemExit(f"no .flac files found in {tar_path}")
        for f in tqdm(flacs, desc=f"convert {shard}"):
            target = bg_out / (f.stem + ".wav")
            if not target.exists():
                try:
                    _write_wav(target, _to_16k_int16(f))
                except Exception as exc:  # noqa: BLE001
                    print(f"   skipped {f.name}: {exc}", flush=True)
        shutil.rmtree(ex_dir, ignore_errors=True)
        if not p.get("keep"):
            try:
                os.remove(tar_path)
            except OSError:
                pass
        done_flag.write_text("ok\n")

    # --- pre-computed negative + validation features (large) ---
    feat_dir = wd / "features"
    feat_dir.mkdir(exist_ok=True)
    api = HfApi()
    for fname in (VALIDATION_FILE, ACAV_FILE):
        target = feat_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        try:
            info = api.get_paths_info(HF_FEATURES_REPO, [fname], repo_type="dataset")
            size_gb = info[0].size / 1e9
            print(f"\n[data] {fname}: {size_gb:,.1f} GB (resumable; re-run if interrupted) ...", flush=True)
        except Exception:  # noqa: BLE001
            print(f"\n[data] {fname} (large, resumable; re-run if interrupted) ...", flush=True)
        _retry(lambda: hf_hub_download(HF_FEATURES_REPO, fname, repo_type="dataset", local_dir=str(feat_dir)),
               f"{fname} download")
    print("\n[data] all data ready", flush=True)


def w_synth(p: dict) -> None:
    """Tiny synthetic stand-ins for the big downloads, used by --self-test only."""
    import numpy as np
    wd = Path(p["workdir"])
    rng = np.random.RandomState(0)
    (wd / "mit_rirs").mkdir(exist_ok=True)
    for i in range(4):
        n = 4800
        ir = rng.randn(n) * np.exp(-np.linspace(0, 8, n))
        ir[0] = 1.0
        ir = ir / np.abs(ir).max() * 0.9
        _write_wav(wd / "mit_rirs" / f"rir_{i}.wav", (ir * 32767).astype(np.int16))
    (wd / "audioset_16k").mkdir(exist_ok=True)
    t = np.arange(16000 * 10) / 16000.0
    for i in range(6):
        noise = rng.randn(len(t)) * 0.05
        tone = 0.1 * np.sin(2 * np.pi * (200 + 150 * i) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        _write_wav(wd / "audioset_16k" / f"bg_{i}.wav", ((noise + tone) * 32767).astype(np.int16))
    (wd / "features").mkdir(exist_ok=True)
    np.save(wd / "features" / ACAV_FILE, rng.rand(3000, 16, 96).astype(np.float16))
    np.save(wd / "features" / VALIDATION_FILE, rng.rand(1500, 96).astype(np.float32))
    print("synthetic self-test data written", flush=True)


def w_config(p: dict) -> None:
    import yaml
    oww = Path(p["oww"])
    wd = Path(p["workdir"])
    cfg = yaml.safe_load((oww / "examples" / "custom_model.yml").read_text())
    cfg.update({
        "model_name": p["model_name"],
        "target_phrase": [p["phrase"]],
        "custom_negative_phrases": list(p["negatives"]),
        "n_samples": int(p["n_samples"]),
        "n_samples_val": int(p["n_val"]),
        "steps": int(p["steps"]),
        "piper_sample_generator_path": p["psg"],
        "output_dir": str(wd / "output"),
        "rir_paths": [str(wd / "mit_rirs")],
        "background_paths": [str(wd / "audioset_16k")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(wd / "features" / VALIDATION_FILE),
        "feature_data_files": {"ACAV100M_sample": str(wd / "features" / ACAV_FILE)},
    })
    if p.get("self_test"):
        cfg["tts_batch_size"] = 10
        cfg["batch_n_per_class"] = {"ACAV100M_sample": 64, "adversarial_negative": 16, "positive": 16}
        cfg["max_negative_weight"] = 100
    Path(p["out"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"config written: {p['out']}", flush=True)


def w_verify(p: dict) -> None:
    import glob
    import numpy as np
    from openwakeword.model import Model

    def load(path, framework):
        m = Model(wakeword_models=[path], inference_framework=framework)
        return m, list(m.models.keys())[0]

    def peak(m, key, wav):
        scores = m.predict_clip(wav)
        if hasattr(m, "reset"):
            m.reset()
        return max(s[key] for s in scores)

    pos = sorted(glob.glob(os.path.join(p["clips_dir"], "positive_test", "*.wav")))[:25]
    neg = sorted(glob.glob(os.path.join(p["clips_dir"], "negative_test", "*.wav")))[:25]
    for label, path, framework in (("ONNX", p["onnx"], "onnx"), ("TFLite", p.get("tflite"), "tflite")):
        if not path:
            continue
        m, key = load(path, framework)
        print(f"[{label}] loaded in openWakeWord as '{key}'")
        if pos and neg:
            ps = [peak(m, key, f) for f in pos]
            ns = [peak(m, key, f) for f in neg]
            print(f"[{label}] clean wake-word clips : mean peak score {np.mean(ps):.2f}, "
                  f"{sum(s > 0.5 for s in ps)}/{len(ps)} above 0.5")
            print(f"[{label}] similar-sounding other phrases: mean peak score {np.mean(ns):.2f}, "
                  f"{sum(s > 0.5 for s in ns)}/{len(ns)} above 0.5")
    print("(Sanity check on synthetic clips only: clean wake-word clips should score high; near-miss phrases "
          "should score lower, though not always zero. Real-world accuracy depends on your microphone/room - "
          "test live and tune the threshold.)")


def w_tflite(p: dict) -> None:
    """Runs in the separate tflite environment."""
    import subprocess as sp
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    from ai_edge_litert.interpreter import Interpreter

    out = Path(p["out"])
    out.mkdir(parents=True, exist_ok=True)
    exe = str(Path(sys.executable).parent / "onnx2tf")

    # onnx2tf treats a 3-D (1, 16, 96) input as channels-first and silently turns it into (1, 96, 16), which
    # openWakeWord cannot feed. Workaround: give the ONNX graph a (1, 96, 16) input followed by a Transpose,
    # so that after onnx2tf's own transposition the TFLite input is (1, 16, 96) again.
    model = onnx.load(p["onnx"])
    graph = model.graph
    inp = graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 3:
        raise SystemExit(f"unexpected model input shape {dims}")
    new_name = "oww_input_transposed"
    graph.node.insert(0, helper.make_node("Transpose", [new_name], [inp.name], perm=[0, 2, 1],
                                          name="oww_input_transpose"))
    graph.input.remove(inp)
    graph.input.insert(0, helper.make_tensor_value_info(new_name, TensorProto.FLOAT, [dims[0], dims[2], dims[1]]))
    onnx.checker.check_model(model)
    wrapped = out / "wrapped_for_tflite.onnx"
    onnx.save(model, str(wrapped))

    sp.run([exe, "-i", str(wrapped), "-o", str(out / "o2t")], check=True)
    candidates = sorted((out / "o2t").glob("*_float32.tflite")) or sorted((out / "o2t").glob("*.tflite"))
    if not candidates:
        raise SystemExit("onnx2tf produced no .tflite file")

    # Never ship a conversion we have not checked: same input layout and same outputs as the ONNX model.
    interp = Interpreter(model_path=str(candidates[0]))
    interp.allocate_tensors()
    tin, tout = interp.get_input_details()[0], interp.get_output_details()[0]
    if [int(d) for d in tin["shape"]] != dims:
        raise SystemExit(f"tflite input shape {tin['shape'].tolist()} does not match expected {dims}")
    ref = ort.InferenceSession(p["onnx"], providers=["CPUExecutionProvider"])
    ref_in = ref.get_inputs()[0].name
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(50):
        x = rng.rand(*dims).astype(np.float32)
        want = float(ref.run(None, {ref_in: x})[0].ravel()[0])
        interp.set_tensor(tin["index"], x)
        interp.invoke()
        got = float(interp.get_tensor(tout["index"]).ravel()[0])
        worst = max(worst, abs(want - got))
    print(f"tflite vs onnx: worst absolute output difference over 50 random inputs = {worst:.2e}", flush=True)
    if worst > 1e-3:
        raise SystemExit("the converted tflite model does not match the ONNX model; not using it")

    shutil.copy2(candidates[0], out / f"{p['name']}.tflite")
    print(f"tflite written: {out / (p['name'] + '.tflite')}", flush=True)


WORKERS = {"gpucheck": w_gpucheck, "preview": w_preview, "fetch": w_fetch, "synth": w_synth,
           "config": w_config, "verify": w_verify, "tflite": w_tflite}

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_worker":
        WORKERS[sys.argv[2]](json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
    else:
        main()

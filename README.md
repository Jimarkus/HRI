# HRI
HRI Book pusher project repo: say "Mister Bookshelf" and what you feel like reading, and the
system picks a book from the shelf and tells the robot where it is.

| Folder | What it is |
|---|---|
| [audio pipeline/](audio%20pipeline/) | Wake word "Mister Bookshelf" -> record -> fast transcription -> text |
| [Project-Alex/](https://github.com/Alwyn-Miller/Project-Alex) | Vision (git submodule): OAK-D Lite camera photo -> YOLO finds each book -> PaddleOCR reads its spine |
| [combination/](combination/) | Joins the two: request text -> recommended book -> where the camera found it on the shelf |

## Setup
```
git clone --recurse-submodules https://github.com/Jimarkus/HRI.git
cd HRI
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r combination/requirements.txt
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
```
The last two lines matter: `ultralytics` pulls in `opencv-python`, which clashes with the
`opencv-contrib-python` build that PaddleOCR needs. Only one of the two can be installed.
Already cloned without `--recurse-submodules`? Run `git submodule update --init`.
Project-Alex stores one 876 MB model with Git LFS; it is not needed here, so set
`GIT_LFS_SKIP_SMUDGE=1` before cloning to skip it.

## Run
```
python combination/run.py                 # scan the last shelf photo, then listen
python combination/run.py --camera        # take a new shelf photo with the OAK-D Lite first
python combination/run.py --text "a book about money"   # no microphone needed
```

## How the pieces connect
1. `combination/shelf.py` calls Project-Alex's `spine_detector.detect_books()` (and `camera.capture_bookshelf()`
   with `--camera`). Books come back numbered left to right with their pixel boxes and the text
   read off each spine. The scan runs once at startup and takes about 45 s (OCR on the CPU).
   `locate_books()` matches each spine's text to the titles and authors in `combination/books.json`,
   so the order of the books on the shelf doesn't matter.
2. `audio pipeline/main.py`'s `MisterBookshelf.listen()` returns the spoken request as a string.
3. `combination/recommender.py` compares the request with each book description in
   `combination/books.json` and picks the best fit.
4. `combination/run.py` passes a `Pick` (the book + where the camera found it) to `handle_pick()`,
   which is where the robot's push command goes. If a spine couldn't be read, the book's optional
   `position` in `books.json` (1 = leftmost) is used instead.

Photos taken with `--camera` are saved to `combination/shelf_capture.jpg` and reused by later runs.

## To do
- Replace the placeholder books and descriptions in `combination/books.json` with the real shelf.
  Spell each title and author as printed on the spine, since the OCR text is matched against them.
- Put the robot's push command in `handle_pick()` in `combination/run.py`.

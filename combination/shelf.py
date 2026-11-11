"""Vision side: wraps Project-Alex (camera.py + spine_detector.py) without modifying it.

Project-Alex finds each book in the shelf photo and reads its spine with PaddleOCR.
`locate_books` then matches that spine text to the catalog, so we know which detected
book is which title.

Project-Alex uses paths relative to its own folder ("yolo11n-seg.pt", "images/", "output/"),
so every call into it runs with that folder as the working directory.
"""
from __future__ import annotations

import difflib
import importlib
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALEX_DIR = ROOT / "Project-Alex"
SAMPLE_IMAGE = ALEX_DIR / "images" / "bookshelf.jpg"
CAPTURE_IMAGE = Path(__file__).resolve().parent / "shelf_capture.jpg"  # photos taken with --camera
ANNOTATED_IMAGE = ALEX_DIR / "output" / "detected_books.jpg"

MIN_MATCH = 0.3        # share of a book's title/author words that must be found on the spine
WORD_SIMILARITY = 0.8  # how close an OCR word must be to count ("PEBSPECTIVE" ~ "perspective")
_IGNORED_WORDS = {"the", "and", "for", "with", "from"}


@contextmanager
def _in_alex_dir():
    if not (ALEX_DIR / "spine_detector.py").is_file():
        raise FileNotFoundError(
            f"{ALEX_DIR} is empty. Fetch the submodule with: git submodule update --init")
    previous = Path.cwd()
    os.chdir(ALEX_DIR)
    if str(ALEX_DIR) not in sys.path:
        sys.path.append(str(ALEX_DIR))  # appended, so it never shadows the audio pipeline's main.py
    try:
        yield
    finally:
        os.chdir(previous)


def default_image() -> Path:
    """The last photo taken with --camera, or Project-Alex's sample photo if there is none."""
    return CAPTURE_IMAGE if CAPTURE_IMAGE.is_file() else SAMPLE_IMAGE


def scan_shelf(use_camera: bool = False, image_path: str | Path | None = None) -> list[dict]:
    """Detect the books on the shelf and read their spines.

    use_camera=True opens the OAK-D Lite preview (press S to take the photo, Q to cancel).
    Otherwise `image_path` is used, defaulting to `default_image()`.

    Returns Project-Alex's detections sorted left to right. Each dict has `book_id`
    (1 = leftmost), `recognized_text` (OCR of the spine), the pixel box `x1,y1,x2,y2`,
    `relative_x`/`relative_y` (0-1 position of the book's centre in the image),
    `horizontal_position`, `vertical_position`, `confidence`.
    """
    image = Path(image_path).resolve() if image_path else default_image()
    with _in_alex_dir():
        if use_camera:
            camera = importlib.import_module("camera")  # needs depthai + the OAK-D plugged in
            # Save into combination/ instead of over Project-Alex's tracked sample photo.
            camera.IMAGE_PATH = CAPTURE_IMAGE
            captured = camera.capture_bookshelf()
            import cv2
            cv2.destroyAllWindows()
            if captured is None:
                raise RuntimeError("No image captured from the camera.")
            image = Path(captured).resolve()
        detector = importlib.import_module("spine_detector")
        return detector.detect_books(image)


# ---------------------------------------------------------------------------------------
# Matching spine text to catalog books
# ---------------------------------------------------------------------------------------
@dataclass
class Location:
    detection: dict   # the Project-Alex detection for this book
    found_by: str     # "ocr" = spine text matched; "position" = fallback to books.json position
    match: float      # OCR match score 0-1 (0 for the position fallback)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 3 and w not in _IGNORED_WORDS]


def spine_match_score(spine_text: str, title: str, author: str = "") -> float:
    """Share of the book's title + author words that appear (approximately) in the spine text."""
    spine, wanted = _words(spine_text), _words(f"{title} {author}")
    if not spine or not wanted:
        return 0.0
    found = 0.0
    for word in wanted:
        best = max(difflib.SequenceMatcher(None, word, s).ratio() for s in spine)
        if best >= WORD_SIMILARITY:
            found += best
    return found / len(wanted)


def locate_books(detections: list[dict], books: list) -> dict[int, Location]:
    """Work out which detection is which catalog book. Returns {index into `books`: Location}.

    1. OCR: every (book, detection) pair is scored on spine text; the best pairs are taken
       first and each book/detection is used once.
    2. Fallback: a book the OCR couldn't place uses its books.json `position` (1 = leftmost),
       if it has one and that detection wasn't already claimed by another title.
    """
    pairs = sorted(
        ((spine_match_score(d.get("recognized_text") or "", b.title, b.author), bi, di)
         for bi, b in enumerate(books) for di, d in enumerate(detections)),
        reverse=True)
    located: dict[int, Location] = {}
    used: set[int] = set()
    for score, bi, di in pairs:
        if score < MIN_MATCH:
            break
        if bi not in located and di not in used:
            located[bi] = Location(detections[di], "ocr", score)
            used.add(di)
    for bi, book in enumerate(books):
        if bi in located or getattr(book, "position", None) is None:
            continue
        for di, d in enumerate(detections):
            if d["book_id"] == book.position and di not in used:
                located[bi] = Location(d, "position", 0.0)
                used.add(di)
    return located


def describe_scan(detections: list[dict], books: list, located: dict[int, Location]) -> str:
    """Human-readable table of what the camera saw and which catalog book each spine is."""
    by_detection = {loc.detection["book_id"]: (books[bi], loc) for bi, loc in located.items()}
    lines = []
    for d in detections:
        text = (d.get("recognized_text") or "").strip() or "(no text read)"
        if d["book_id"] in by_detection:
            book, loc = by_detection[d["book_id"]]
            how = f"match {loc.match:.2f}" if loc.found_by == "ocr" else "by books.json position, text not matched"
            verdict = f"{book.title} ({how})"
        else:
            verdict = "NOT IN CATALOG"
        lines.append(f"  book {d['book_id']} at x={d['relative_x']:.0%}: \"{text[:60]}\" -> {verdict}")
    missing = [b.title for bi, b in enumerate(books) if bi not in located]
    if missing:
        lines.append(f"  WARNING: in books.json but not found on the shelf: {'; '.join(missing)}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recommender import load_catalog
    catalog = load_catalog()
    found = scan_shelf(use_camera="--camera" in sys.argv)
    print(f"{len(found)} books detected (annotated image: {ANNOTATED_IMAGE})")
    print(describe_scan(found, catalog, locate_books(found, catalog)))

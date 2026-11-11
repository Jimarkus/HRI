"""Mister Bookshelf, whole system: voice -> recommendation -> where the book is on the shelf.

    python run.py                    # scan the last shelf photo, then listen for "Mister Bookshelf"
    python run.py --camera           # take a fresh shelf photo with the OAK-D Lite first
    python run.py --text "a book about money"   # skip the microphone (for testing)
    python run.py --no-vision        # skip the shelf scan; shelf position comes from books.json only

Flow:
  1. shelf.py        Project-Alex finds each book in the shelf image and reads its spine (OCR);
                     the spine text is matched to the titles/authors in books.json
  2. audio pipeline  waits for "Mister Bookshelf", records the request, transcribes it
  3. recommender.py  picks the catalog book (books.json) that best fits the request
  4. the pick is returned with where the camera found that book -> where to push

The scan takes about 45 s (the OCR runs on the CPU), once at startup.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audio pipeline"))

from recommender import Book, Recommendation, build_recommender, load_catalog  # noqa: E402
from shelf import ANNOTATED_IMAGE, Location, describe_scan, locate_books, scan_shelf  # noqa: E402


@dataclass
class Pick:
    """Everything the robot needs to act on one request."""
    request: str
    book: Book
    score: float
    location: Location | None  # where the camera found this book (.detection has the pixel box), or None
    recommendation: Recommendation


class BookshelfSystem:
    def __init__(self, recommender: str = "embedding", vision: bool = True, camera: bool = False,
                 image: str | None = None):
        self.books = load_catalog()
        self.detections: list[dict] = []
        self.locations: dict[int, Location] = {}  # index into self.books -> where the camera found it
        if vision:
            print("Scanning the shelf and reading the spines (about 45 s)...")
            self.detections = scan_shelf(use_camera=camera, image_path=image)
            self.locations = locate_books(self.detections, self.books)
            print(f"  camera sees {len(self.detections)} books (annotated image: {ANNOTATED_IMAGE})")
            print(describe_scan(self.detections, self.books, self.locations))
        print(f"Loading the {recommender} recommender for {len(self.books)} books...")
        self.recommender = build_recommender(recommender, self.books)

    def pick(self, request: str) -> Pick:
        rec = self.recommender.recommend(request)
        location = self.locations.get(self.books.index(rec.book))
        return Pick(request, rec.book, rec.score, location, rec)


def handle_pick(pick: Pick) -> None:
    """Hook for the robot: replace the prints with the command that pushes the book out."""
    book, loc = pick.book, pick.location
    author = f" by {book.author}" if book.author else ""
    print(f'>>> Request: "{pick.request}"')
    print(f"    Recommend: {book.title}{author}  (score {pick.score:.2f})")
    if loc:
        d = loc.detection
        how = "spine text read by the camera" if loc.found_by == "ocr" else "books.json position, spine not read"
        print(f"    Location:  book {d['book_id']} from the left, centre at x={d['relative_x']:.0%} "
              f"y={d['relative_y']:.0%} of the image, box ({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")
        print(f"               found by: {how}")
    elif book.position is not None:
        print(f"    Location:  book {book.position} from the left (from books.json, not seen by the camera)")
    else:
        print("    Location:  unknown (not seen by the camera)")
    if len(pick.recommendation.ranking) > 1:
        runner_up, score = pick.recommendation.ranking[1]
        print(f"    Runner-up: {runner_up.title} ({score:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="use this request instead of listening on the microphone")
    ap.add_argument("--recommender", default="embedding", choices=["embedding", "hybrid"],
                    help="embedding = description similarity (default); hybrid = embeddings shortlist, Laya picks")
    ap.add_argument("--camera", action="store_true", help="capture a new shelf photo with the OAK-D Lite")
    ap.add_argument("--image", help="shelf photo to scan (default: the last --camera photo, "
                                    "else Project-Alex/images/bookshelf.jpg)")
    ap.add_argument("--no-vision", action="store_true", help="skip the shelf scan")
    ap.add_argument("--once", action="store_true", help="exit after one request")
    ap.add_argument("--model", help="faster-whisper model (default: small.en on GPU, base.en on CPU)")
    ap.add_argument("--wake", default="auto", choices=["auto", "oww", "whisper"])
    ap.add_argument("--mic", type=int, help="PyAudio input device index")
    ap.add_argument("--no-sound", action="store_true", help="disable the beeps")
    args = ap.parse_args()

    system = BookshelfSystem(args.recommender, not args.no_vision, args.camera, args.image)
    if args.text:
        handle_pick(system.pick(args.text))
        return

    from main import MisterBookshelf  # audio pipeline/main.py
    bot = MisterBookshelf(args.model, args.wake, mic=args.mic, sound=not args.no_sound)
    try:
        while True:
            handle_pick(system.pick(bot.listen()))
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        bot.close()


if __name__ == "__main__":
    main()

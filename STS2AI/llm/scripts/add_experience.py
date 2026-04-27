"""Append compact lessons to the local LLM experience library."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.experience_library import (  # noqa: E402
    DEFAULT_EXPERIENCE_PATH,
    ExperienceEntry,
    append_experience,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", required=True, help="Comma-separated tags, e.g. vulnerable,attack,boss.")
    parser.add_argument("--when", required=True, dest="applies_when")
    parser.add_argument("--advice", required=True)
    parser.add_argument("--avoid", default="")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--path", default=str(DEFAULT_EXPERIENCE_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = ExperienceEntry(
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        applies_when=args.applies_when.strip(),
        advice=args.advice.strip(),
        avoid=args.avoid.strip(),
        source=args.source.strip(),
        confidence=args.confidence,
    )
    append_experience([entry], Path(args.path))
    print(f"[experience] appended -> {Path(args.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

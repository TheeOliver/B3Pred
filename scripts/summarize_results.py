#!/usr/bin/env python3
"""Flatten metric JSON files produced by the B3Pred CLI into a CSV."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file or directory to scan recursively")
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()

    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.json"))
    rows = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        rows.append({"source": str(path), **payload})

    if not rows:
        raise SystemExit(f"No metric JSON objects found under {args.input}")

    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} result rows to {args.output}")


if __name__ == "__main__":
    main()

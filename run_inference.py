#!/usr/bin/env python3
"""Run inference using the project's `evaluate_essay` API.

Usage examples:
  python run_inference.py --text "My essay text here" --prompt "Write about X"
  python run_inference.py --file sample.txt --weights kaggle/working/best_model.weights.h5 -o out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from essay_pipeline import evaluate_essay


def main() -> None:
    p = argparse.ArgumentParser(description="Run essay inference using the local model pipeline")
    p.add_argument("--text", "-t", help="Inline essay text")
    p.add_argument("--file", "-f", help="Path to a text file containing the essay")
    p.add_argument("--prompt", "-p", help="Prompt/context for the essay", default=None)
    p.add_argument("--weights", "-w", help="Path to weights HDF5 file", default="kaggle/working/best_model.weights.h5")
    p.add_argument("--rubric", help="Optional rubric string", default=None)
    p.add_argument("--output", "-o", help="Write JSON result to this file", default=None)

    args = p.parse_args()

    if not args.text and not args.file:
        p.error("Provide --text or --file")

    if args.file:
        essay_path = Path(args.file)
        if not essay_path.exists():
            raise SystemExit(f"File not found: {essay_path}")
        text = essay_path.read_text(encoding="utf-8")
    else:
        text = args.text

    evaluation = evaluate_essay(
        text=text,
        prompt=args.prompt,
        rubric=args.rubric,
        reference_texts=None,
        weights_path=args.weights,
    )

    out = evaluation.to_dict()
    out_json = json.dumps(out, indent=2)

    if args.output:
        Path(args.output).write_text(out_json, encoding="utf-8")
        print(f"Wrote results to {args.output}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

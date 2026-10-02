from __future__ import annotations

import argparse
from collections import defaultdict

from dualflow.io import read_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/e2/derived/arm_results.jsonl")
    p.add_argument("--split", default="test")
    return p.parse_args()


def main():
    args = parse_args()
    rows = [r for r in read_jsonl(args.input) if r["split"] == args.split]
    print(f"split={args.split} episodes={len(rows)}")
    print("arm\tunsafe\tfalse_reject\texecute")
    for arm in ("single", "repeated", "grounded"):
        unsafe = sum(r["arms"][arm]["unsafe_execution"] for r in rows)
        fr = sum(r["arms"][arm]["false_reject"] for r in rows)
        ex = sum(r["arms"][arm]["execute"] for r in rows)
        print(f"{arm}\t{unsafe}/{len(rows)}\t{fr}/{len(rows)}\t{ex}/{len(rows)}")


if __name__ == "__main__":
    main()

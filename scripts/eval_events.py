#!/usr/bin/env python3
"""
Evaluate detected bat-contact events against a hand-labelled ground truth.

Computes precision / recall / F1 by matching each detected event to a
ground-truth contact within a temporal tolerance (default ±1.0 s). Optionally
scores shot-name accuracy on the true-positive matches.

Ground-truth CSV columns (header required):
    clip,timestamp_sec,shot_name
  - one row per REAL bat-contact you observe in the clip
  - `shot_name` optional (leave blank if you only label presence)

Usage:
    python scripts/eval_events.py \
        --gt docs/eval/ground_truth.csv \
        --events events_clip1.json events_clip2.json events_clip3.json \
        --clips clip1 clip2 clip3 \
        --tol 1.0
"""
import argparse, csv, json, os
from collections import defaultdict


def load_gt(path):
    gt = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("timestamp_sec", "").strip():
                continue
            gt[row["clip"].strip()].append(
                dict(t=float(row["timestamp_sec"]),
                     shot=(row.get("shot_name") or "").strip())
            )
    return gt


def load_events(path):
    return [dict(t=float(e["timestamp_sec"]), shot=e.get("shot_name"))
            for e in json.load(open(path))]


def match(det, gt, tol):
    """Greedy nearest-neighbour matching within tol seconds."""
    gt_used = [False] * len(gt)
    tp, fp = [], []
    shot_correct = 0
    for d in sorted(det, key=lambda x: x["t"]):
        best, bj = tol + 1e9, -1
        for j, g in enumerate(gt):
            if gt_used[j]:
                continue
            dt = abs(d["t"] - g["t"])
            if dt < best:
                best, bj = dt, j
        if bj >= 0 and best <= tol:
            gt_used[bj] = True
            tp.append((d, gt[bj], best))
            if gt[bj]["shot"] and d["shot"] and gt[bj]["shot"].lower() == d["shot"].lower():
                shot_correct += 1
        else:
            fp.append(d)
    fn = [g for j, g in enumerate(gt) if not gt_used[j]]
    return tp, fp, fn, shot_correct


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--clips", nargs="+", required=True,
                    help="clip key per events file, matching the gt 'clip' column")
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    assert len(args.events) == len(args.clips), "events and clips must align"

    gt = load_gt(args.gt)
    TP = FP = FN = SC = 0
    print(f"{'clip':<10}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'rec':>8}{'F1':>8}")
    print("-" * 46)
    for ev_path, clip in zip(args.events, args.clips):
        det = load_events(ev_path)
        tp, fp, fn, sc = match(det, gt.get(clip, []), args.tol)
        p, r, f1 = prf(len(tp), len(fp), len(fn))
        print(f"{clip:<10}{len(tp):>4}{len(fp):>4}{len(fn):>4}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
        TP += len(tp); FP += len(fp); FN += len(fn); SC += sc
    p, r, f1 = prf(TP, FP, FN)
    print("-" * 46)
    print(f"{'OVERALL':<10}{TP:>4}{FP:>4}{FN:>4}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
    if TP:
        print(f"\nShot-name accuracy on true positives: {SC}/{TP} = {100*SC/TP:.1f}% "
              f"(only counts GT rows that specified a shot_name)")
    print(f"\nMatching tolerance: ±{args.tol:.1f} s")


if __name__ == "__main__":
    main()

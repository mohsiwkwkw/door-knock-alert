"""Honest error bars for the key comparisons in results/stress_results.json.

The same 40 knock clips are reused across volumes and backgrounds, so mixtures are not
independent. This resamples KNOCK CLIPS with replacement (cluster bootstrap) and reports the
mean difference in mixture recall with a 95% interval. An interval that includes 0 means the
difference could be chance.

Batches in this file
  1  Tools and the three comparisons to test
  2  main: for each comparison, resample the knock clips 4,000 times
  3  Save the results next to the input file

Run: .venv/bin/python -I src/bootstrap_contrasts.py --results results/stress_results.json
"""
import argparse
import json

import numpy as np


# ===== BATCH 1 | Tools and the three comparisons to test =====
# (the detector we hope is better, the detector we compare it with, a plain-English label)
PAIRS = [
    ("single (aug-trained)", "single (clean-trained)", "training with mixed sounds (single-stage)"),
    ("two-wave (aug-trained)", "single (aug-trained)", "two-wave rule on top of the mixed-sound model"),
    ("two-wave (clean-trained)", "single (clean-trained)", "two-wave rule on top of the clean-trained model"),
]


def main():
    # ===== BATCH 2 | main: for each comparison, resample the knock clips 4,000 times =====
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    r = json.load(open(a.results, encoding="utf-8"))
    rng = np.random.default_rng(0)
    out = {}
    for tkey, block in r["operating_points"].items():
        out[tkey] = {}
        print(f"\n== {tkey} ==")
        for better, base, label in PAIRS:
            hb = np.array(block[better]["mixture_hits"])   # rows: knock index, snr, hit
            ha = np.array(block[base]["mixture_hits"])
            assert (hb[:, 0] == ha[:, 0]).all() and (hb[:, 1] == ha[:, 1]).all()
            knocks = np.unique(hb[:, 0])
            for scope, mask in (("all volumes", np.ones(len(hb), bool)),
                                ("quiet knock (-10 dB)", hb[:, 1] == -10),
                                ("loud knock (+10 dB)", hb[:, 1] == 10)):
                diff = (hb[:, 2] - ha[:, 2]).astype(float)
                per_knock = {k: diff[(hb[:, 0] == k) & mask].mean() for k in knocks}   # one number per knock clip
                vals = np.array([per_knock[k] for k in knocks])
                boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(4000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                out[tkey][f"{label} | {scope}"] = dict(diff=float(vals.mean()), ci95=[float(lo), float(hi)])
                flag = "clear" if lo > 0 or hi < 0 else "could be chance"
                print(f"{label} | {scope}: {vals.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {flag}")

    # ===== BATCH 3 | Save the results next to the input file =====
    json.dump(out, open(a.results.replace(".json", "_contrasts.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

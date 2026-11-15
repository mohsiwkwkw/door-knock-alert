"""Simple scorecard on ESC-50 fold 5 (400 clips: 8 knocks and 392 other sounds).

This is the held-out test set the demo models never trained on, so other models tested on the
same 400 clips can be compared in the same format. A clip counts as "knock detected" if the
model raises at least one alert anywhere in it, and as a "false alert" if a non-knock clip raises
at least one alert. Uses the demo models and thresholds from build_demo_model.py.

Run: .venv/bin/python -I src/fold5_report.py --data data/esc50_raw/ESC-50-master --demo demo --out results
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_stress_test as rst  # noqa: E402


def metrics(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return dict(clips=n, accuracy=(tp + tn) / n, knocks_detected=tp, knocks_total=tp + fn, knocks_missed=fn,
                recall=rec, false_alerts=fp, non_knock_clips=fp + tn, false_alert_rate=fp / (fp + tn),
                precision=prec, f1=f1, recall_ci95=rst.wilson(tp, tp + fn), false_alert_rate_ci95=rst.wilson(fp, fp + tn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rst.DATA = a.data
    with open(os.path.join(a.data, "meta", "esc50.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["fold"]) == 5]
    with ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        recs = list(ex.map(rst.clip_worker, [r["filename"] for r in rows], chunksize=10))
    models = joblib.load(os.path.join(a.demo, "models.joblib"))
    cfg = json.load(open(os.path.join(a.demo, "config.json"), encoding="utf-8"))
    out = {"test_set": "ESC-50 fold 5", "clips": len(rows),
           "knocks": sum(r["category"] == rst.KNOCK for r in rows)}
    for label, key, thr in (("clean-trained", "clean", cfg["threshold_clean"]),
                            ("mixed-sound-trained", "aug", cfg["threshold_aug"])):
        tp = fn = fp = tn = 0
        missed, flagged = [], []
        for r, rc in zip(rows, recs):
            X, t = rst.windows_of(rc["F"], rc["r"], rc["a0"], rc["b0"])
            hit = len(rst.alerts_model(models[key].decision_function(X), t, thr)) > 0
            if r["category"] == rst.KNOCK:
                tp += hit; fn += not hit
                if not hit:
                    missed.append(r["filename"])
            else:
                fp += hit; tn += not hit
                if hit:
                    flagged.append((r["filename"], r["category"]))
        out[label] = dict(metrics(tp, fn, fp, tn), missed_knock_clips=missed, falsely_flagged_clips=flagged)
    n_non = out["clean-trained"]["non_knock_clips"]
    out["always say no"] = metrics(0, out["knocks"], 0, n_non)
    json.dump(out, open(os.path.join(a.out, "fold5_report.json"), "w", encoding="utf-8"), indent=1)
    print(f"{'':28s}{'accuracy':>10s}{'knocks':>9s}{'false alerts':>14s}{'precision':>11s}{'F1':>7s}")
    for label in ("always say no", "clean-trained", "mixed-sound-trained"):
        m = out[label]
        print(f"{label:28s}{m['accuracy'] * 100:9.1f}%{m['knocks_detected']:>5d}/{m['knocks_total']:<3d}"
              f"{m['false_alerts']:>8d}/{m['non_knock_clips']:<5d}{m['precision'] * 100:10.1f}%{m['f1'] * 100:6.1f}%")
    for label in ("clean-trained", "mixed-sound-trained"):
        m = out[label]
        print(f"\n{label}: knocks caught {m['recall'] * 100:.1f}% (95% CI {m['recall_ci95'][0] * 100:.0f}% to {m['recall_ci95'][1] * 100:.0f}%), "
              f"false alerts {m['false_alert_rate'] * 100:.2f}% of non-knock clips")
        print("  missed knock clips:", m["missed_knock_clips"])
        print("  falsely flagged clips:", m["falsely_flagged_clips"])


if __name__ == "__main__":
    main()

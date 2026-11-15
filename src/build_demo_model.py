"""Train the two models the demo uses, on ESC-50 folds 1 to 4 only.

Fold 5 is never used for training, so the demo's example clips (all taken from fold 5) are
clips the models have never seen. Two models, same recipe as run_stress_test.py:
  clean - trained on clean knock windows versus windows of every other sound
  aug   - the same plus knock windows mixed into home sounds at random volumes

Alert thresholds come from the pooled matched-false-alert operating points in
results/stress_results.json (about 5 false alerts per hour for the models; about 20 per hour
for the loudness alarm, which cannot get lower). The demo models are single retrained
instances, so their exact false-alert rate can differ a little from the pooled figure.

Run: .venv/bin/python -I src/build_demo_model.py --data data/esc50_raw/ESC-50-master --results results --out demo
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_stress_test as rst  # noqa: E402

TRAIN_FOLDS = {1, 2, 3, 4}
DEMO_FOLD = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rst.DATA = a.data
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.data, "meta", "esc50.csv"), newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    fname = [m["filename"] for m in meta]
    fold = np.array([int(m["fold"]) for m in meta])
    cat = np.array([m["category"] for m in meta])
    is_knock = cat == rst.KNOCK
    is_home = np.array([c in rst.HOME for c in cat])
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(6, mp_context=ctx) as ex:
        recs = list(ex.map(rst.clip_worker, fname, chunksize=20))

    rng = np.random.default_rng(0)
    pos, neg = [], []
    for i, rc in enumerate(recs):
        if fold[i] not in TRAIN_FOLDS:
            continue
        if is_knock[i]:
            for o in rst.knock_onsets(rc["on"], rc["a0"], rc["b0"]):
                for off in (-6, -4, -2):
                    s = max(rc["a0"], min(o + off, rc["b0"] - rst.WIN_FR))
                    e = min(rc["F"].shape[0], s + rst.WIN_FR)
                    pos.append(rst.win_feats(rc["F"], rc["r"], s, e))
        else:
            X, _ = rst.windows_of(rc["F"], rc["r"], rc["a0"], rc["b0"])
            for j in rng.choice(len(X), size=min(rst.NEG_PER_CLIP, len(X)), replace=False):
                neg.append(X[j])
    pos = np.array(pos, dtype="float32"); neg = np.array(neg, dtype="float32")

    knock_idx = [i for i in np.nonzero(is_knock)[0] if fold[i] in TRAIN_FOLDS]
    home_by_fold = {f: np.nonzero(is_home & (fold == f) & ~is_knock)[0] for f in TRAIN_FOLDS}
    specs, owners = [], []
    for ki in knock_idx:
        for _ in range(rst.AUG_MIX_PER_KNOCK):
            bf = int(rng.choice([f for f in TRAIN_FOLDS if f != fold[ki]]))
            bi = int(rng.choice(home_by_fold[bf]))
            specs.append((fname[ki], fname[bi], float(rng.uniform(-10, 15)), int(rng.integers(1 << 30))))
            owners.append(ki)
    with ProcessPoolExecutor(6, mp_context=ctx) as ex:
        mixes = list(ex.map(rst.mix_worker, specs, chunksize=10))
    aug = []
    for mr, ki in zip(mixes, owners):
        kr = recs[ki]
        shift = int(round((mr["off"] - mr["kl"]) / rst.HOP))
        T = mr["F"].shape[0]
        for o in rst.knock_onsets(kr["on"], kr["a0"], kr["b0"]):
            o2 = o + shift
            if 0 <= o2 < T - 8:
                s = max(0, min(o2 - 4, T - rst.WIN_FR))
                aug.append(rst.win_feats(mr["F"], mr["r"], s, min(T, s + rst.WIN_FR)))
    aug = np.array(aug, dtype="float32")
    print(f"training windows: {len(pos)} knock, {len(neg)} other, {len(aug)} mixed-sound knock", flush=True)

    def mk():
        return rst.make_pipeline(rst.StandardScaler(), rst.SVC(C=2.0, kernel="rbf", class_weight="balanced"))

    y1 = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clean = mk().fit(np.vstack([pos, neg]), y1)
    augm = mk().fit(np.vstack([pos, neg, aug]), np.r_[y1, np.ones(len(aug))])

    ops = json.load(open(os.path.join(a.results, "stress_results.json"), encoding="utf-8"))["operating_points"]
    cfg = dict(
        threshold_clean=ops["5_false_alerts_per_hour"]["single (clean-trained)"]["setting"]["thi"],
        threshold_aug=ops["5_false_alerts_per_hour"]["single (aug-trained)"]["setting"]["thi"],
        loudness_level_db=ops["20_false_alerts_per_hour"]["loudness alarm (clean only)"]["setting"]["param"],
        train_folds=sorted(TRAIN_FOLDS), demo_fold=DEMO_FOLD,
        demo_knocks=[fname[i] for i in np.nonzero(is_knock & (fold == DEMO_FOLD))[0]],
        demo_home={c: [fname[i] for i in np.nonzero((cat == c) & (fold == DEMO_FOLD))[0]] for c in sorted(rst.HOME)},
    )
    joblib.dump({"clean": clean, "aug": augm}, os.path.join(a.out, "models.joblib"))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), indent=1)
    print("saved models and config;", {k: cfg[k] for k in ("threshold_clean", "threshold_aug", "loudness_level_db")})


if __name__ == "__main__":
    main()

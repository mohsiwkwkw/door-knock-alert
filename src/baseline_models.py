"""Steps 3 and 4 (first pass): train and test knock-vs-rest models with leave-one-fold-out.

Input: results/features.npz from build_features.py (one 132-number row per whole clip).

For each held-out ESC-50 fold, models train on the other four folds. The alert threshold is
chosen using only the training folds (inner cross-validation scores), never the test fold.
Two operating points are reported:
  - "maxF1": the threshold with the best F1 on training scores.
  - "recall90": the strictest threshold that still catches 90% of training knocks.
Reported per operating point: recall (knocks caught), false-alarm rate on non-knock clips,
precision. Also average precision and ROC AUC per fold, and which non-knock classes are
flagged most often. Accuracy is deliberately not reported (always saying "no" scores 98%).

Batches in this file
  1  Tools and the three candidate models
  2  Helpers: score a model, get honest training scores, choose thresholds, count results
  3  main, part A: for every held-out fold, train, choose thresholds, test
  4  main, part B: save the results

Usage:
    .venv/bin/python -I src/baseline_models.py --features results/features.npz --out results
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ===== BATCH 1 | Tools and the three candidate models =====
# Each entry builds a fresh, untrained model. class_weight="balanced" makes the rare knock
# examples count more, because there are 49 non-knock clips for every knock clip.
MODELS = {
    "logreg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)),
    "svm_rbf": lambda: make_pipeline(
        StandardScaler(), SVC(C=2.0, kernel="rbf", class_weight="balanced")),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", random_state=0, n_jobs=-1),
}


# ===== BATCH 2 | Helpers =====
def score(model, X):
    """One 'how knock-like?' number per row (the SVM and logreg give a distance; the forest a probability)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def oof_scores(make, X, y, n=4):
    """Out-of-fold scores: every training clip is scored by a model that never saw that clip."""
    s = np.zeros(len(y))
    for tr, va in StratifiedKFold(n, shuffle=True, random_state=0).split(X, y):
        s[va] = score(make().fit(X[tr], y[tr]), X[va])
    return s


def thr_max_f1(s, y):
    """The alert level with the best F1 on these scores."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def thr_recall(s, y, target=0.9):
    """The strictest alert level that still catches `target` (90%) of the knocks."""
    pos = np.sort(s[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def counts(s, y, thr):
    """Knocks caught, knocks missed, false alerts, and quiet clips correctly left alone."""
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fn, fp, tn


def main():
    # ===== BATCH 3 | main, part A: for every held-out fold, train, choose thresholds, test =====
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.features, allow_pickle=False)
    X, y, fold, cat = d["X"], d["y"], d["fold"], d["category"]
    secs = d["trimmed_seconds"]
    report = {"diagnostic_auc_trimmed_duration_alone": float(roc_auc_score(y, secs))}
    clip_rows = []

    for name, make in MODELS.items():
        agg = {"maxF1": [0, 0, 0, 0], "recall90": [0, 0, 0, 0]}
        ap_list, auc_list = [], []
        flagged = Counter()
        negatives = Counter(cat[y == 0])
        for f in sorted(set(fold.tolist())):
            tr, te = fold != f, fold == f                        # train on four folds, test on the fifth
            s_tr = oof_scores(make, X[tr], y[tr])
            thrs = {"maxF1": thr_max_f1(s_tr, y[tr]), "recall90": thr_recall(s_tr, y[tr])}
            model = make().fit(X[tr], y[tr])
            s_te = score(model, X[te])
            ap_list.append(float(average_precision_score(y[te], s_te)))
            auc_list.append(float(roc_auc_score(y[te], s_te)))
            for op, t in thrs.items():
                c = counts(s_te, y[te], t)
                agg[op] = [x + z for x, z in zip(agg[op], c)]
            for c_name, yy, ss in zip(cat[te], y[te], s_te):
                if yy == 0 and ss >= thrs["maxF1"]:
                    flagged[c_name] += 1
            for fn_, c_name, yy, ss in zip(d["filename"][te], cat[te], y[te], s_te):
                clip_rows.append((name, fn_, int(f), c_name, int(yy), float(ss)))
        res = {"mean_average_precision": float(np.mean(ap_list)), "mean_roc_auc": float(np.mean(auc_list))}
        for op, (tp, fn, fp, tn) in agg.items():
            res[op] = {
                "recall": tp / (tp + fn), "false_alarm_rate_non_knock": fp / (fp + tn),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "knocks_caught": tp, "knocks_missed": fn, "false_alarms": fp, "non_knock_clips": fp + tn,
            }
        res["most_flagged_non_knock_classes_maxF1"] = [
            (c, flagged[c], negatives[c]) for c, _ in flagged.most_common(8)
        ]
        report[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "most_flagged_non_knock_classes_maxF1"}, indent=1))

    # ===== BATCH 4 | main, part B: save the results =====
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "baselines.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(a.out, "clip_scores.csv"), "w", encoding="utf-8") as fh:
        fh.write("model,filename,fold,category,is_knock,score\n")
        for r in clip_rows:
            fh.write(",".join(map(str, r)) + "\n")
    print("diagnostic AUC from trimmed duration alone:", report["diagnostic_auc_trimmed_duration_alone"])


if __name__ == "__main__":
    main()

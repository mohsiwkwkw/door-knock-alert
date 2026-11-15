"""Compare the pre-built YAMNet sound model with our own SVM, in the same stress test.

Three ways to use YAMNet (all single-stage; YAMNet reports a score every 0.48 s):
  yamnet_knock         zero-shot: YAMNet's own "Knock" class score
  yamnet_knock_or_door zero-shot: the larger of its "Knock" and "Door" class scores
  yamnet+logreg        modify the pre-built model: a logistic regression on its 1024-number
                       embeddings, trained leave-one-fold-out, clean-trained and mixed-sound-trained

Same rules as run_stress_test.py: operating points matched at 5 and 20 false alerts per hour of
ESC-50 non-knock sound; mixtures of held-out knocks and held-out home sounds at +10 to -10 dB;
a hit counts only if an alert falls inside the knock's time span. Training frames: every valid
frame of a knock clip counts as a knock frame (weak labels).

Run (main environment): .venv/bin/python -I src/run_yamnet_eval.py --results results
"""
import argparse
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KNOCK = "door_wood_knock"
HOME = {"mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine",
        "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking", "footsteps", "clapping",
        "coughing", "sneezing", "breathing", "brushing_teeth", "drinking_sipping", "laughing",
        "snoring", "crying_baby", "pouring_water", "water_drops", "toilet_flush", "dog", "cat"}
SNRS = [10, 5, 0, -5, -10]
TARGETS = [5.0, 20.0]


def merge(times, refr=1.0):
    out = []
    for x in sorted(times):
        if not out or x - out[-1] >= refr:
            out.append(float(x))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def split_by_owner(owner, n):
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(n + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = a.results
    meta = json.load(open(os.path.join(R, "mixture_meta.json"), encoding="utf-8"))
    cat = np.array(meta["categories"]); fold = np.array(meta["folds"]); n = len(cat)
    is_knock = cat == KNOCK
    is_home = np.array([c in HOME for c in cat])
    names = list(np.load(os.path.join(R, "yamnet_class_names.npy")))
    k_i, d_i = names.index("Knock"), names.index("Door")

    C = np.load(os.path.join(R, "yam_clean.npz"))
    Au = np.load(os.path.join(R, "yam_aug.npz"))
    Te = np.load(os.path.join(R, "yam_test.npz"))
    c_slices = split_by_owner(C["owner"], n)
    a_slices = split_by_owner(Au["owner"], len(meta["aug"]))
    t_slices = split_by_owner(Te["owner"], len(meta["test"]))
    lo_s, hi_s = C["lo_s"], C["hi_s"]

    neg_idx = np.nonzero(~is_knock)[0]
    knock_idx = np.nonzero(is_knock)[0]
    home_set = set(int(i) for i in neg_idx if is_home[i])
    hours_all = float(sum(hi_s[i] - lo_s[i] for i in neg_idx) / 3600)
    hours_home = float(sum(hi_s[i] - lo_s[i] for i in home_set) / 3600)

    # ---- training frames for the logistic regression ----
    valid = C["valid"]; emb = C["emb"]
    pos_rows, neg_rows = [], []
    for i in range(n):
        rows_i = c_slices[i][valid[c_slices[i]]]
        (pos_rows if is_knock[i] else neg_rows).append((i, rows_i))
    # augmentation positives: mixture frames overlapping the knock span
    aug_pos = []                                   # (knock fold, bg fold, row indices)
    for j, m in enumerate(meta["aug"]):
        rows_j = a_slices[j]
        t = Au["ctime"][rows_j]
        s0, s1 = m["off"] / 44100.0, (m["off"] + m["L"]) / 44100.0
        ov = np.minimum(t - 0.48 + 0.96, s1) - np.maximum(t - 0.48, s0)
        aug_pos.append((fold[m["knock_index"]], m["bg_fold"], rows_j[ov >= 0.25]))

    def fit_lr(test_fold, use_aug):
        pr = np.concatenate([r for i, r in pos_rows if fold[i] != test_fold])
        nr = np.concatenate([r for i, r in neg_rows if fold[i] != test_fold])
        X = [emb[pr], emb[nr]]; y = [np.ones(len(pr)), np.zeros(len(nr))]
        if use_aug:
            ar = [r for kf, bf, r in aug_pos if kf != test_fold and bf != test_fold and len(r)]
            if ar:
                ar = np.concatenate(ar)
                X.append(Au["emb"][ar]); y.append(np.ones(len(ar)))
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
        return mdl.fit(np.vstack(X), np.concatenate(y))

    # ---- frame scores per detector ----
    clean_sc, test_sc = {}, {}
    zs = {"yamnet_knock": lambda S: S[:, k_i].astype("float32"),
          "yamnet_knock_or_door": lambda S: np.maximum(S[:, k_i], S[:, d_i]).astype("float32")}
    for name, fn in zs.items():
        clean_sc[name] = [fn(C["scores"][c_slices[i]]) for i in range(n)]
        test_sc[name] = [fn(Te["scores"][t_slices[j]]) for j in range(len(meta["test"]))]
    for name, aug in (("yamnet+logreg (clean-trained)", False), ("yamnet+logreg (mixed-sound-trained)", True)):
        clean_sc[name] = [None] * n
        test_sc[name] = [None] * len(meta["test"])
        for f in range(1, 6):
            mdl = fit_lr(f, aug)
            for i in np.nonzero(fold == f)[0]:
                clean_sc[name][i] = mdl.decision_function(emb[c_slices[i]])
            for j, m in enumerate(meta["test"]):
                if m["fold"] == f:
                    test_sc[name][j] = mdl.decision_function(Te["emb"][t_slices[j]])
        print("trained", name, flush=True)

    ctime_c = [C["ctime"][c_slices[i]] for i in range(n)]
    valid_c = [valid[c_slices[i]] for i in range(n)]
    ctime_t = [Te["ctime"][t_slices[j]] for j in range(len(meta["test"]))]

    def clip_alerts(name, i, thr):
        s, t, v = clean_sc[name][i], ctime_c[i], valid_c[i]
        return merge(list(t[(s >= thr) & v]))

    results = {"hours_all_negative_audio": hours_all, "hours_home_negative_audio": hours_home, "operating_points": {}}
    curves = {}
    for name in clean_sc:
        negv = np.concatenate([clean_sc[name][i][valid_c[i]] for i in neg_idx])
        grid = np.unique(np.quantile(negv, np.linspace(0.90, 0.99995, 160)))
        rows = []
        for thr in grid:
            rec = float(np.mean([len(clip_alerts(name, i, thr)) > 0 for i in knock_idx]))
            cnt = {int(i): len(clip_alerts(name, i, thr)) for i in neg_idx}
            rows.append(dict(thr=float(thr), recall=rec, fa_all=sum(cnt.values()) / hours_all,
                             fa_home=sum(v for i, v in cnt.items() if i in home_set) / hours_home))
        curves[name] = rows
    results["curves"] = curves

    def mix_hits(name, thr):
        res, hl = {}, []
        for snr in SNRS:
            hit = tot = 0
            for j, m in enumerate(meta["test"]):
                if m["snr"] != snr:
                    continue
                lo, hi = m["off"] / 44100.0 - 0.2, (m["off"] + m["L"]) / 44100.0 + 1.0
                s, t = test_sc[name][j], ctime_t[j]
                al = merge(list(t[s >= thr]))
                h = any(lo <= x <= hi for x in al)
                hit += h; tot += 1
                hl.append((int(m["knock_index"]), int(snr), int(h)))
            res[str(snr)] = dict(hits=int(hit), total=int(tot), recall=hit / tot, ci95=wilson(int(hit), int(tot)))
        return res, hl

    for target in TARGETS:
        block = {}
        for name, rows in curves.items():
            ok = [r for r in rows if r["fa_all"] <= target]
            best = max(ok, key=lambda r: (r["recall"], -r["fa_all"])) if ok else min(rows, key=lambda r: r["fa_all"])
            res, hl = mix_hits(name, best["thr"])
            block[name] = dict(setting=best, target_met=bool(best["fa_all"] <= target), clean_recall=best["recall"],
                               clean_recall_ci95=wilson(int(round(best["recall"] * 40)), 40),
                               false_alerts_per_hour_all=best["fa_all"], false_alerts_per_hour_home=best["fa_home"],
                               mixture_recall_by_snr=res, mixture_hits=hl)
        results["operating_points"][f"{target:g}_false_alerts_per_hour"] = block

    # ---- bring in our SVM numbers for a side-by-side table and error bars ----
    svm = json.load(open(os.path.join(R, "stress_results.json"), encoding="utf-8"))["operating_points"]
    rng = np.random.default_rng(0)
    contrasts = {}
    for tkey, block in results["operating_points"].items():
        print(f"\n== {tkey} ==")
        table = [("SVM single-stage (clean-trained)", svm[tkey]["single (clean-trained)"]),
                 ("SVM single-stage (mixed-sound-trained)", svm[tkey]["single (aug-trained)"])] + \
                [(k, v) for k, v in block.items()]
        for label, e in table:
            mr = e["mixture_recall_by_snr"]
            flag = "" if e.get("target_met", True) else " [TARGET NOT MET]"
            print(f"{label:42s} clean {e['clean_recall']:.2f} | FA/h {e['false_alerts_per_hour_all']:.1f}{flag} | mixed "
                  + " ".join(f"{s}dB:{mr[s]['recall']:.2f}" for s in mr))
        contrasts[tkey] = {}
        base = np.array(svm[tkey]["single (aug-trained)"]["mixture_hits"])
        for name in ("yamnet+logreg (mixed-sound-trained)", "yamnet_knock"):
            hb = np.array(block[name]["mixture_hits"])
            assert (hb[:, 0] == base[:, 0]).all() and (hb[:, 1] == base[:, 1]).all()
            knocks = np.unique(hb[:, 0])
            diff = (hb[:, 2] - base[:, 2]).astype(float)
            vals = np.array([diff[hb[:, 0] == k].mean() for k in knocks])
            boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(4000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            contrasts[tkey][name + " minus SVM (mixed-sound-trained)"] = dict(diff=float(vals.mean()), ci95=[float(lo), float(hi)])
            print(f"  {name} minus our mixed-sound SVM, all volumes: {vals.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  + ("" if lo > 0 or hi < 0 else "  could be chance"))
    results["contrasts_vs_svm"] = contrasts
    with open(os.path.join(R, "yamnet_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("\nsaved results/yamnet_results.json")


if __name__ == "__main__":
    main()

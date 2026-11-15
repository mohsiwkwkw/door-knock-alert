"""Step 1 (clean): check and describe ESC-50 using only the Python standard library.

Why standard library only: this machine has no numpy or pip yet, and this step needs
none of them. The extracted dataset is treated as untrusted data: this script only
reads it, never runs anything from inside it.

Usage (from the project folder, not from inside the data folder):
    python3 -I src/inspect_data.py --data data/esc50_raw/ESC-50-master --out results
"""
import argparse
import csv
import json
import math
import os
import warnings
import wave
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402  (present in Python 3.12, removed in 3.13)

WIN = 441  # 10 ms at 44.1 kHz


def read_meta(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dbfs(x):
    return 20 * math.log10(x / 32768.0) if x > 0 else -120.0


def analyze_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    info = {"channels": ch, "sampwidth": sw, "rate": rate, "frames": n}
    if ch != 1 or sw != 2:
        info["unsupported"] = True
        return info
    nwin = n // WIN
    zero = []
    for i in range(nwin):
        chunk = raw[i * WIN * 2:(i + 1) * WIN * 2]
        zero.append(audioop.max(chunk, 2) == 0)
    lead = 0
    while lead < nwin and zero[lead]:
        lead += 1
    trail = 0
    while trail < nwin - lead and zero[nwin - 1 - trail]:
        trail += 1
    info.update(
        rms=audioop.rms(raw, 2),
        peak=audioop.max(raw, 2),
        zero_frac=sum(zero) / nwin if nwin else 0.0,
        lead_frac=lead / nwin if nwin else 0.0,
        trail_frac=trail / nwin if nwin else 0.0,
    )
    return info


def auc(scores_pos, scores_neg):
    """Rank-based AUC (Mann-Whitney), ties get average rank."""
    allv = sorted([(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rpos = sum(r for r, (_, lab) in zip(ranks, allv) if lab == 1)
    npos, nneg = len(scores_pos), len(scores_neg)
    return (rpos - npos * (npos + 1) / 2) / (npos * nneg)


def best_threshold(rows, key):
    """Pick the threshold on `key` that maximizes F1 for knock vs rest."""
    vals = sorted({r[key] for r in rows})
    best = (0.0, vals[0])
    for t in vals:
        tp = sum(1 for r in rows if r["knock"] and r[key] >= t)
        fp = sum(1 for r in rows if not r["knock"] and r[key] >= t)
        fn = sum(1 for r in rows if r["knock"] and r[key] < t)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = read_meta(os.path.join(a.data, "meta", "esc50.csv"))
    checks = {}
    checks["rows"] = len(meta)
    checks["folds"] = dict(sorted(Counter(r["fold"] for r in meta).items()))
    checks["classes"] = len({r["category"] for r in meta})
    per_class = Counter(r["category"] for r in meta)
    checks["clips_per_class_min_max"] = [min(per_class.values()), max(per_class.values())]
    src_folds = defaultdict(set)
    for r in meta:
        src_folds[r["src_file"]].add(r["fold"])
    checks["source_files_split_across_folds"] = sum(1 for v in src_folds.values() if len(v) > 1)
    knock_cats = sorted({r["category"] for r in meta if "knock" in r["category"]})
    checks["knock_categories"] = knock_cats

    rows = []
    bad_format = []
    missing = []
    for r in meta:
        p = os.path.join(a.data, "audio", r["filename"])
        if not os.path.exists(p):
            missing.append(r["filename"])
            continue
        info = analyze_wav(p)
        if info.get("unsupported") or info["rate"] != 44100 or info["frames"] != 220500:
            bad_format.append((r["filename"], info))
        if info.get("unsupported"):
            continue
        rows.append({
            "filename": r["filename"], "fold": int(r["fold"]), "category": r["category"],
            "knock": r["category"] in knock_cats[:1] if knock_cats else False,
            "src_file": r["src_file"],
            **{k: info[k] for k in ("rms", "peak", "zero_frac", "lead_frac", "trail_frac")},
        })
    checks["missing_files"] = len(missing)
    checks["unexpected_audio_format"] = len(bad_format)

    # Which knock class is "the" door knock? Prefer the one named *_knock
    knock_name = next((c for c in knock_cats), None)
    knock_rows = [r for r in rows if r["category"] == knock_name]
    other_rows = [r for r in rows if r["category"] != knock_name]
    for r in rows:
        r["knock"] = r["category"] == knock_name

    knock_per_fold = dict(sorted(Counter(r["fold"] for r in knock_rows).items()))
    knock_sources = len({r["src_file"] for r in knock_rows})

    # Padding (digital silence) by class
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["category"]].append(r["zero_frac"])
    class_pad = {c: sum(v) / len(v) for c, v in by_class.items()}
    pad_rank = sorted(class_pad.items(), key=lambda kv: -kv[1])

    results = {
        "checks": checks,
        "knock_class": knock_name,
        "knock_clips": len(knock_rows),
        "knock_per_fold": knock_per_fold,
        "knock_distinct_source_files": knock_sources,
        "knock_mean_padded_fraction": class_pad.get(knock_name),
        "all_other_mean_padded_fraction": sum(r["zero_frac"] for r in other_rows) / len(other_rows),
        "knock_clips_with_any_padding": sum(1 for r in knock_rows if r["zero_frac"] > 0),
        "other_clips_with_any_padding": sum(1 for r in other_rows if r["zero_frac"] > 0),
        "padding_rank_top10": pad_rank[:10],
        "knock_padding_rank": [c for c, _ in pad_rank].index(knock_name) + 1,
        "auc_padding_alone": auc([r["zero_frac"] for r in knock_rows], [r["zero_frac"] for r in other_rows]),
        "auc_rms_loudness": auc([r["rms"] for r in knock_rows], [r["rms"] for r in other_rows]),
        "auc_peak_loudness": auc([r["peak"] for r in knock_rows], [r["peak"] for r in other_rows]),
        "knock_rms_dbfs_mean": sum(dbfs(r["rms"]) for r in knock_rows) / len(knock_rows),
        "others_rms_dbfs_mean": sum(dbfs(r["rms"]) for r in other_rows) / len(other_rows),
    }

    # Loudness-threshold baseline, leave-one-fold-out, clip level, on peak level
    folds = sorted({r["fold"] for r in rows})
    base = []
    tp = fp = fn = tn = 0
    for f in folds:
        train = [r for r in rows if r["fold"] != f]
        test = [r for r in rows if r["fold"] == f]
        t = best_threshold(train, "peak")
        ftp = sum(1 for r in test if r["knock"] and r["peak"] >= t)
        ffn = sum(1 for r in test if r["knock"] and r["peak"] < t)
        ffp = sum(1 for r in test if not r["knock"] and r["peak"] >= t)
        ftn = sum(1 for r in test if not r["knock"] and r["peak"] < t)
        base.append({"fold": f, "threshold_peak": t, "tp": ftp, "fn": ffn, "fp": ffp, "tn": ftn})
        tp += ftp; fn += ffn; fp += ffp; tn += ftn
    results["loudness_baseline_leave_one_fold_out"] = {
        "per_fold": base,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_alarm_rate_on_non_knock_clips": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "always_say_no_accuracy": (tn + fp) / (tp + fn + tn + fp),
    }

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(a.out, "clip_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

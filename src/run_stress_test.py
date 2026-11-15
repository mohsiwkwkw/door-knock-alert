"""Fixed-window knock detector, two-wave decision, and mixing stress test (ESC-50).

What this does
  1. Cleaning rule: trim digital-silence padding; features are gain-invariant (no c0, no
     absolute loudness) and computed on FIXED 0.5 s windows, so clip length cannot leak.
  2. Trains an SVM on windows (knock windows versus windows of every other sound), leave-one-fold-out.
     Two training conditions: "clean" and "aug" (knock windows mixed into home sounds at random volumes).
  3. Detectors compared (all scan a clip with 0.5 s windows every ~0.1 s):
       loudness  - alert when loudness passes a level (clean test only; ESC-50 levels are arbitrary)
       rhythm    - NON-AI rule: two prominent loudness peaks within 1.5 s
       single    - alert when one window scores above T_high
       two_wave  - alert when one window scores above T_high OR two peaks above T_low = T_high - d
                   fall 0.2 to 2.0 s apart
  4. Clean test: recall on knock clips versus false alerts per hour of non-knock sound.
  5. Mixing stress test: held-out knocks mixed into held-out HOME sounds at several volumes
     (SNR +10 to -10 dB). A detection counts only if an alert falls inside the knock's time span.

Operating points are set at a matched false-alert rate on the pooled held-out clean negatives,
the same way for every detector. ESC-50 non-knock clips are dense with sound events, so
"false alerts per hour" here is per hour of ESC-50 sound, not per hour of a quiet home.

Run from the doorknock folder:
    .venv/bin/python -I src/run_stress_test.py --data data/esc50_raw/ESC-50-master --out results
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR, HOP, NFFT = 44100, 512, 2048
SEC = HOP / SR                      # seconds per frame
WIN_FR, STRIDE_FR = 43, 9           # 0.5 s windows, hop about 0.1 s
KNOCK = "door_wood_knock"
HOME = {"mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine",
        "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking", "footsteps", "clapping",
        "coughing", "sneezing", "breathing", "brushing_teeth", "drinking_sipping", "laughing",
        "snoring", "crying_baby", "pouring_water", "water_drops", "toilet_flush", "dog", "cat"}
SNRS = [10, 5, 0, -5, -10]
NEG_PER_CLIP = 5
AUG_MIX_PER_KNOCK = 6
TEST_BG_PER_KNOCK_SNR = 6
TARGETS = [5.0, 20.0]               # false alerts per hour of ESC-50 sound
DATA = None


# ---------- audio and features ----------
def trim_bounds(y):
    n = len(y) // 441
    w = np.abs(y[: n * 441]).reshape(n, 441).max(axis=1)
    nz = np.nonzero(w > 0)[0]
    if len(nz) == 0:
        return 0, 441
    return int(nz[0] * 441), int((nz[-1] + 1) * 441)


def frame_feats(y):
    st = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=st ** 2, sr=SR, n_mels=64)
    logmel = librosa.power_to_db(mel)
    mf = librosa.feature.mfcc(S=logmel, n_mfcc=21)[1:]          # drop c0: no absolute loudness
    dl = librosa.feature.delta(mf, width=9)
    cen = librosa.feature.spectral_centroid(S=st, sr=SR)
    bw = librosa.feature.spectral_bandwidth(S=st, sr=SR)
    ro = librosa.feature.spectral_rolloff(S=st, sr=SR)
    zc = librosa.feature.zero_crossing_rate(y, frame_length=NFFT, hop_length=HOP)
    rms = librosa.feature.rms(S=st, frame_length=NFFT)[0]
    onset = librosa.onset.onset_strength(S=logmel, sr=SR, hop_length=HOP)
    T = min(a.shape[-1] for a in (mf, dl, cen, bw, ro, zc, rms, onset))
    F = np.vstack([mf[:, :T], dl[:, :T], cen[:, :T], bw[:, :T], ro[:, :T], zc[:, :T]]).T.astype("float32")
    rms_db = (20 * np.log10(rms[:T] + 1e-9)).astype("float32")
    return F, rms_db, onset[:T].astype("float32")


def win_feats(F, r, a, b):
    seg, rr = F[a:b], r[a:b]
    rel = rr - rr.max()
    lin = 10 ** (rr / 20)
    crest = lin.max() / (lin.mean() + 1e-12)
    return np.concatenate([seg.mean(0), seg.std(0), seg.max(0), [rel.mean(), rel.std(), rel.min(), crest]])


def window_spans(a0, b0, T):
    b0 = min(b0, T)
    if b0 - a0 <= WIN_FR:
        return [(a0, max(b0, min(a0 + 8, T)))]
    return [(s, s + WIN_FR) for s in range(a0, b0 - WIN_FR + 1, STRIDE_FR)]


def windows_of(F, r, a0, b0):
    spans = window_spans(a0, b0, F.shape[0])
    X = np.array([win_feats(F, r, s, e) for s, e in spans], dtype="float32")
    t = np.array([(s + e) / 2.0 * SEC for s, e in spans])
    return X, t


def knock_onsets(onset_env, a0, b0):
    on = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="frames")
    on = [int(o) for o in on if a0 <= o < b0]
    return on if on else [int(a0 + np.argmax(onset_env[a0:b0]))]


def rms_lin(x):
    return float(np.sqrt(np.mean(np.square(x.astype("float64")))) + 1e-12)


def make_mix(k, kl, kh, bg, bl, bh, snr_db, rng):
    n = len(bg)
    b = bg.astype("float64").copy()
    floor = rng.normal(0.0, 10 ** (-55.0 / 20), size=n)
    pad = np.ones(n, dtype=bool)
    pad[bl:bh] = False
    b[pad] = floor[pad]
    kn = k[kl:kh].astype("float64")
    L = min(len(kn), n)
    off = int(rng.integers(0, n - L + 1))
    placed = np.zeros(n)
    placed[off:off + L] = kn[:L]
    gain = rms_lin(b[bl:bh]) / rms_lin(kn[:L]) * 10 ** (snr_db / 20)
    return (b + gain * placed).astype("float32"), off, L


# ---------- workers (forked processes) ----------
def clip_worker(fname):
    y, sr = sf.read(os.path.join(DATA, "audio", fname), dtype="float32")
    lo, hi = trim_bounds(y)
    F, r, on = frame_feats(y)
    a0, b0 = lo // HOP, min(F.shape[0], -(-hi // HOP))
    return dict(F=F, r=r, on=on, a0=a0, b0=b0)


def mix_worker(spec):
    kf, bf, snr, seed = spec
    rng = np.random.default_rng(seed)
    k, _ = sf.read(os.path.join(DATA, "audio", kf), dtype="float32")
    bg, _ = sf.read(os.path.join(DATA, "audio", bf), dtype="float32")
    kl, kh = trim_bounds(k)
    bl, bh = trim_bounds(bg)
    mix, off, L = make_mix(k, kl, kh, bg, bl, bh, snr, rng)
    F, r, on = frame_feats(mix)
    return dict(F=F, r=r, on=on, off=off, L=L, kl=kl,
                audio16=resample_poly(mix, 160, 441).astype("float32"))


# ---------- detectors ----------
def merge(times, refr=1.0):
    out = []
    for x in sorted(times):
        if not out or x - out[-1] >= refr:
            out.append(float(x))
    return out


def alerts_model(s, t, thi, tlo=None):
    cand = list(t[s >= thi])
    if tlo is not None:
        idx, _ = find_peaks(s, height=tlo, distance=2)
        pt = t[idx]
        for j in range(1, len(pt)):
            d = pt[j] - pt[:j]
            if np.any((d >= 0.2) & (d <= 2.0)):
                cand.append(pt[j])
    return merge(cand)


def alerts_rhythm(r, a0, b0, prom):
    env = r[a0:b0]
    if len(env) < 3:
        return []
    idx, _ = find_peaks(env, prominence=prom, distance=13)
    pt = (a0 + idx) * SEC
    cand = [pt[j] for j in range(1, len(pt)) if pt[j] - pt[j - 1] <= 1.5]
    return merge(cand)


def alerts_loud(r, a0, b0, level):
    fr = np.nonzero(r[a0:b0] >= level)[0]
    return merge([(a0 + f) * SEC for f in fr])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mixtures-only", action="store_true",
                    help="build the mixtures, save their 16 kHz audio for the YAMNet step, then stop")
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    with open(os.path.join(a.data, "meta", "esc50.csv"), newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    n = len(meta)
    fname = [m["filename"] for m in meta]
    fold = np.array([int(m["fold"]) for m in meta])
    cat = np.array([m["category"] for m in meta])
    is_knock = cat == KNOCK
    is_home = np.array([c in HOME for c in cat])
    ctx = mp.get_context("fork")

    # 1. frame features for every clip
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        recs = list(ex.map(clip_worker, fname, chunksize=20))
    print(f"[{time.time() - t0:.0f}s] frame features for {n} clips", flush=True)

    # 2. clean windows: per-clip test windows, training positives and negatives
    rng = np.random.default_rng(0)
    clip_win = []
    pos_clean, neg_train = [], []
    for i, rc in enumerate(recs):
        X, t = windows_of(rc["F"], rc["r"], rc["a0"], rc["b0"])
        clip_win.append((X, t))
        if is_knock[i]:
            for o in knock_onsets(rc["on"], rc["a0"], rc["b0"]):
                for off in (-6, -4, -2):
                    s = max(rc["a0"], min(o + off, rc["b0"] - WIN_FR))
                    e = min(rc["F"].shape[0], s + WIN_FR)
                    pos_clean.append((fold[i], win_feats(rc["F"], rc["r"], s, e)))
        else:
            k = min(NEG_PER_CLIP, len(X))
            for j in rng.choice(len(X), size=k, replace=False):
                neg_train.append((fold[i], X[j]))
    pf = np.array([p[0] for p in pos_clean]); px = np.array([p[1] for p in pos_clean], dtype="float32")
    nf = np.array([p[0] for p in neg_train]); nx = np.array([p[1] for p in neg_train], dtype="float32")
    print(f"[{time.time() - t0:.0f}s] training windows: {len(px)} knock, {len(nx)} other", flush=True)

    # 3. mixtures: augmentation (training) and stress test (held-out)
    knock_idx = np.nonzero(is_knock)[0]
    home_by_fold = {f: np.nonzero(is_home & (fold == f) & ~is_knock)[0] for f in range(1, 6)}
    aug_specs, aug_meta = [], []
    for ki in knock_idx:
        for m in range(AUG_MIX_PER_KNOCK):
            other = [f for f in range(1, 6) if f != fold[ki]]
            bf = int(rng.choice(other))
            bi = int(rng.choice(home_by_fold[bf]))
            snr = float(rng.uniform(-10, 15))
            aug_specs.append((fname[ki], fname[bi], snr, int(rng.integers(1 << 30))))
            aug_meta.append((ki, bf))
    test_specs, test_meta = [], []
    for ki in knock_idx:
        f = fold[ki]
        for snr in SNRS:
            for m in range(TEST_BG_PER_KNOCK_SNR):
                bi = int(rng.choice(home_by_fold[f]))
                test_specs.append((fname[ki], fname[bi], float(snr), int(rng.integers(1 << 30))))
                test_meta.append((ki, f, snr))
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
        aug_recs = list(ex.map(mix_worker, aug_specs, chunksize=10))
        test_recs = list(ex.map(mix_worker, test_specs, chunksize=10))
    print(f"[{time.time() - t0:.0f}s] {len(aug_recs)} augmentation and {len(test_recs)} test mixtures", flush=True)
    if a.mixtures_only:
        np.save(os.path.join(a.out, "mix_aug_16k.npy"), np.stack([m["audio16"] for m in aug_recs]))
        np.save(os.path.join(a.out, "mix_test_16k.npy"), np.stack([m["audio16"] for m in test_recs]))
        with open(os.path.join(a.out, "mixture_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(
                aug=[dict(knock_index=int(k), bg_fold=int(b), off=int(m["off"]), L=int(m["L"]), kl=int(m["kl"]))
                     for (k, b), m in zip(aug_meta, aug_recs)],
                test=[dict(knock_index=int(k), fold=int(f), snr=int(s), off=int(m["off"]), L=int(m["L"]))
                      for (k, f, s), m in zip(test_meta, test_recs)],
                filenames=fname, folds=[int(x) for x in fold], categories=[str(c) for c in cat]), fh)
        print("saved 16 kHz mixtures and metadata; stopping (--mixtures-only)")
        return
    for m in aug_recs + test_recs:
        m.pop("audio16", None)

    aug_pos, aug_pos_kf, aug_pos_bf = [], [], []
    for mr, (ki, bf) in zip(aug_recs, aug_meta):
        kr = recs[ki]
        shift = int(round((mr["off"] - mr["kl"]) / HOP))
        T = mr["F"].shape[0]
        for o in knock_onsets(kr["on"], kr["a0"], kr["b0"]):
            o2 = o + shift
            if not (0 <= o2 < T - 8):
                continue
            s = max(0, min(o2 - 4, T - WIN_FR))
            aug_pos.append(win_feats(mr["F"], mr["r"], s, min(T, s + WIN_FR)))
            aug_pos_kf.append(fold[ki]); aug_pos_bf.append(bf)
    aug_pos = np.array(aug_pos, dtype="float32"); aug_pos_kf = np.array(aug_pos_kf); aug_pos_bf = np.array(aug_pos_bf)
    print(f"[{time.time() - t0:.0f}s] augmentation windows: {len(aug_pos)}", flush=True)

    test_win = [windows_of(mr["F"], mr["r"], 0, mr["F"].shape[0]) for mr in test_recs]

    # 4. train per fold, score clean clips and test mixtures
    def mk():
        return make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", class_weight="balanced", cache_size=1000))

    scores = {"clean": [None] * n, "aug": [None] * n}
    mix_scores = {"clean": [None] * len(test_recs), "aug": [None] * len(test_recs)}
    for f in range(1, 6):
        trp, trn = pf != f, nf != f
        Xc = np.vstack([px[trp], nx[trn]]); yc = np.r_[np.ones(trp.sum()), np.zeros(trn.sum())]
        m_clean = mk().fit(Xc, yc)
        am = (aug_pos_kf != f) & (aug_pos_bf != f)
        Xa = np.vstack([Xc, aug_pos[am]]); ya = np.r_[yc, np.ones(am.sum())]
        m_aug = mk().fit(Xa, ya)
        for name, mdl in (("clean", m_clean), ("aug", m_aug)):
            for i in np.nonzero(fold == f)[0]:
                scores[name][i] = mdl.decision_function(clip_win[i][0])
            for j, (ki, ff, snr) in enumerate(test_meta):
                if ff == f:
                    mix_scores[name][j] = mdl.decision_function(test_win[j][0])
        print(f"[{time.time() - t0:.0f}s] fold {f} done (train windows clean {len(yc)}, aug {len(ya)})", flush=True)

    # 5. clean-test curves
    neg_idx = np.nonzero(~is_knock)[0]
    hours_all = sum((recs[i]["b0"] - recs[i]["a0"]) * SEC for i in neg_idx) / 3600
    home_neg = [i for i in neg_idx if is_home[i]]
    hours_home = sum((recs[i]["b0"] - recs[i]["a0"]) * SEC for i in home_neg) / 3600

    def curve_model(name, d):
        grid = np.round(np.arange(-0.5, 3.01, 0.05), 3)       # fine steps in SVM score units
        rows = []
        home_set = set(home_neg)
        for thi in grid:
            tlo = None if d is None else thi - d
            rec = np.mean([len(alerts_model(scores[name][i], clip_win[i][1], thi, tlo)) > 0 for i in knock_idx])
            counts = {i: len(alerts_model(scores[name][i], clip_win[i][1], thi, tlo)) for i in neg_idx}
            fa_all = sum(counts.values())
            fa_home = sum(v for i, v in counts.items() if i in home_set)
            rows.append(dict(thi=float(thi), tlo=None if tlo is None else float(tlo), recall=float(rec),
                             fa_all=fa_all / hours_all, fa_home=fa_home / hours_home))
        return rows

    def curve_simple(kind, grid):
        rows = []
        for g in grid:
            fn = (lambda i: alerts_rhythm(recs[i]["r"], recs[i]["a0"], recs[i]["b0"], g)) if kind == "rhythm" \
                else (lambda i: alerts_loud(recs[i]["r"], recs[i]["a0"], recs[i]["b0"], g))
            rec = np.mean([len(fn(i)) > 0 for i in knock_idx])
            fa_all = sum(len(fn(i)) for i in neg_idx)
            fa_home = sum(len(fn(i)) for i in home_neg)
            rows.append(dict(param=float(g), recall=float(rec), fa_all=fa_all / hours_all, fa_home=fa_home / hours_home))
        return rows

    curves = {
        "single_clean": curve_model("clean", None),
        "single_aug": curve_model("aug", None),
        "rhythm": curve_simple("rhythm", list(range(4, 61, 4))),
        "loudness": curve_simple("loudness", list(np.arange(-40, 1, 1.0))),
    }
    for d in (0.5, 1.0, 1.5):
        curves[f"two_wave_clean_d{d}"] = curve_model("clean", d)
        curves[f"two_wave_aug_d{d}"] = curve_model("aug", d)
    print(f"[{time.time() - t0:.0f}s] clean curves done", flush=True)

    def pick(rows, target):
        ok = [r for r in rows if r["fa_all"] <= target]
        if ok:
            return max(ok, key=lambda r: (r["recall"], -r["fa_all"]))
        return min(rows, key=lambda r: r["fa_all"])

    # 6. stress test at matched operating points
    def mix_recall(variant, row):
        res, hl = {}, []
        for snr in SNRS:
            hit = tot = 0
            for j, (ki, f, s_) in enumerate(test_meta):
                if s_ != snr:
                    continue
                mr, t_ = test_recs[j], test_win[j][1]
                lo, hi = mr["off"] / SR - 0.2, (mr["off"] + mr["L"]) / SR + 1.0
                if variant.startswith("rhythm"):
                    al = alerts_rhythm(mr["r"], 0, mr["F"].shape[0], row["param"])
                else:
                    nm = "aug" if "aug" in variant else "clean"
                    al = alerts_model(mix_scores[nm][j], t_, row["thi"], row["tlo"])
                h = any(lo <= x <= hi for x in al)
                hit += h
                tot += 1
                hl.append((int(ki), int(snr), int(h)))
            res[str(snr)] = dict(hits=int(hit), total=int(tot), recall=hit / tot, ci95=wilson(int(hit), int(tot)))
        return res, hl

    families = {
        "single (clean-trained)": ["single_clean"],
        "two-wave (clean-trained)": [f"two_wave_clean_d{d}" for d in (0.5, 1.0, 1.5)],
        "single (aug-trained)": ["single_aug"],
        "two-wave (aug-trained)": [f"two_wave_aug_d{d}" for d in (0.5, 1.0, 1.5)],
        "rhythm rule (no AI)": ["rhythm"],
    }
    report = {"hours_all_negative_audio": hours_all, "hours_home_negative_audio": hours_home,
              "knock_clips": int(len(knock_idx)), "test_mixtures_per_snr": len(test_recs) // len(SNRS),
              "operating_points": {}}
    for target in TARGETS:
        block = {}
        for fam, keys in families.items():
            best = max((pick(curves[k], target) | {"curve": k} for k in keys),
                       key=lambda r: (r["fa_all"] <= target, r["recall"], -r["fa_all"]))
            vname = "rhythm" if fam.startswith("rhythm") else ("aug" if "aug-trained" in fam else "clean")
            rec_by_snr, hits = mix_recall(vname, best)
            block[fam] = dict(setting=best, target_met=bool(best["fa_all"] <= target), clean_recall=best["recall"],
                              clean_recall_ci95=wilson(int(round(best["recall"] * 40)), 40),
                              false_alerts_per_hour_all=best["fa_all"], false_alerts_per_hour_home=best["fa_home"],
                              mixture_recall_by_snr=rec_by_snr, mixture_hits=hits)
        lb = pick(curves["loudness"], target)
        block["loudness alarm (clean only)"] = dict(setting=lb, target_met=bool(lb["fa_all"] <= target), clean_recall=lb["recall"],
                                                    false_alerts_per_hour_all=lb["fa_all"], false_alerts_per_hour_home=lb["fa_home"])
        report["operating_points"][f"{target:g}_false_alerts_per_hour"] = block
    report["curves"] = curves
    with open(os.path.join(a.out, "stress_results.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)

    # 7. summary printout and charts
    for tkey, block in report["operating_points"].items():
        print(f"\n== operating point: {tkey} (set on pooled held-out ESC-50 non-knock sound) ==")
        for fam, e in block.items():
            line = f"{fam:30s} clean recall {e['clean_recall']:.2f} | FA/h all {e['false_alerts_per_hour_all']:.1f} home {e['false_alerts_per_hour_home']:.1f}" + ("" if e["target_met"] else " [TARGET NOT MET]")
            if "mixture_recall_by_snr" in e:
                line += " | mixed recall " + " ".join(f"{s}dB:{v['recall']:.2f}" for s, v in e["mixture_recall_by_snr"].items())
            print(line)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lab in (("single_clean", "single-stage model"), ("single_aug", "single-stage, aug-trained"),
                     ("two_wave_clean_d1.0", "two-wave model (d=1.0)"), ("two_wave_aug_d1.0", "two-wave, aug-trained (d=1.0)"),
                     ("rhythm", "rhythm rule (no AI)"), ("loudness", "loudness alarm")):
        rows = sorted(curves[key], key=lambda r: r["fa_all"])
        ax.plot([max(r["fa_all"], 0.5) for r in rows], [r["recall"] for r in rows], marker=".", label=lab)
    ax.set_xscale("log"); ax.set_xlabel("false alerts per hour of ESC-50 non-knock sound"); ax.set_ylabel("knock clips caught")
    ax.set_title("Clean clips: knocks caught vs false alerts"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "clean_curves.png"), dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, len(TARGETS), figsize=(11, 4.2), sharey=True)
    for ax, (tkey, block) in zip(np.atleast_1d(axs), report["operating_points"].items()):
        for fam, e in block.items():
            if "mixture_recall_by_snr" in e:
                ax.plot(SNRS, [e["mixture_recall_by_snr"][str(s)]["recall"] for s in SNRS], marker="o", label=fam)
        ax.invert_xaxis(); ax.set_xlabel("knock volume relative to home sound (dB; right is quieter)")
        ax.set_title(tkey.replace("_", " ")); ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].set_ylabel("knocks caught"); np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "mixture_recall.png"), dpi=140); plt.close(fig)
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

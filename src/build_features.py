"""Steps 1b and 2 (first pass): clean each clip, then turn the WHOLE clip into one row of numbers.

Cleaning rule (from the data check, see README):
  - ESC-50 clips are padded with digital silence, and padding alone separates knocks from the
    rest (AUC 0.75). So leading and trailing all-zero 10 ms windows are removed before any
    feature is computed.
  - Overall volume is removed by scaling each trimmed clip to a peak of 1, so absolute
    loudness cannot be used. Loudness is tested on purpose later, in the mixing test.
  - Features are length-invariant summaries (mean, std, max over frames). Trimmed duration is
    saved only as a diagnostic and is never a feature.

The output (results/features.npz) feeds baseline_models.py, the first comparison of models.
The better detector (run_stress_test.py) later works on 0.5 s windows instead of whole clips.

Batches in this file
  1  Tools and settings
  2  Cut off the silent padding
  3  Describe one clip with 132 numbers
  4  main: loop over all 2,000 clips and save the table

Usage (run from the doorknock folder, with the project venv's python):
    .venv/bin/python -I src/build_features.py --data data/esc50_raw/ESC-50-master --out results/features.npz
"""
import argparse
import csv
import os

import librosa
import numpy as np
import soundfile as sf


# ===== BATCH 1 | Tools and settings =====
SR = 44100                   # samples per second in every ESC-50 clip
HOP10MS = 441                # 441 samples = 10 ms
KNOCK = "door_wood_knock"    # the ESC-50 category we treat as "a door knock"


# ===== BATCH 2 | Cut off the silent padding =====
def trim_padding(y):
    n = len(y) // HOP10MS
    if n == 0:
        return y
    w = np.abs(y[: n * HOP10MS]).reshape(n, HOP10MS).max(axis=1)   # loudest sample in each 10 ms piece
    nz = np.nonzero(w > 0)[0]                                      # pieces that are not pure silence
    if len(nz) == 0:
        return y[:HOP10MS]
    return y[nz[0] * HOP10MS: (nz[-1] + 1) * HOP10MS]


# ===== BATCH 3 | Describe one clip with 132 numbers =====
def clip_features(y):
    y = y / (np.max(np.abs(y)) + 1e-9)                   # scale so the peak is 1: volume is removed
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)))                # very short clips are padded so the maths works
    kw = dict(sr=SR, n_fft=2048, hop_length=512)         # 2048-sample frames, a new one every 512 samples
    mf = librosa.feature.mfcc(y=y, n_mfcc=20, **kw)      # 20 "tone colour" numbers per frame
    d = librosa.feature.delta(mf, width=3)               # how those 20 numbers change from frame to frame
    cen = librosa.feature.spectral_centroid(y=y, **kw)   # brightness
    bw = librosa.feature.spectral_bandwidth(y=y, **kw)   # how spread out the pitches are
    roll = librosa.feature.spectral_rolloff(y=y, **kw)   # where the high pitches cut off
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    stack = np.vstack([mf, d, cen, bw, roll, zcr])       # 44 rows, one column per frame
    return np.concatenate([stack.mean(axis=1), stack.std(axis=1), stack.max(axis=1)])   # 44 x 3 = 132


def main():
    # ===== BATCH 4 | main: loop over all 2,000 clips and save the table =====
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(os.path.join(a.data, "meta", "esc50.csv"), newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))

    X, y, fold, names, cats, secs = [], [], [], [], [], []
    for i, r in enumerate(meta):
        wav, sr = sf.read(os.path.join(a.data, "audio", r["filename"]), dtype="float32")
        assert sr == SR and wav.ndim == 1, r["filename"]
        t = trim_padding(wav)
        X.append(clip_features(t))
        y.append(int(r["category"] == KNOCK))
        fold.append(int(r["fold"]))
        names.append(r["filename"])
        cats.append(r["category"])
        secs.append(len(t) / SR)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(meta)} clips done", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=np.array(X, dtype="float32"), y=np.array(y), fold=np.array(fold),
             filename=np.array(names), category=np.array(cats), trimmed_seconds=np.array(secs))
    print("saved", a.out, "X shape", np.array(X).shape, "knocks", int(np.sum(y)))


if __name__ == "__main__":
    main()

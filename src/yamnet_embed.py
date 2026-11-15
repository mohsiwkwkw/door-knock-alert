"""Extract YAMNet outputs for every ESC-50 clip and for the saved stress-test mixtures.

Run with the TensorFlow environment (it is separate so the main environment stays untouched):
    TFHUB_CACHE_DIR=$PWD/cache/tfhub .venv-tf/bin/python -I src/yamnet_embed.py \
        --data data/esc50_raw/ESC-50-master --results results

Inputs: results/mix_aug_16k.npy, results/mix_test_16k.npy (written by
`run_stress_test.py --mixtures-only`). Outputs: results/yam_clean.npz, yam_aug.npz, yam_test.npz.

YAMNet is not gain-invariant, so every waveform is scaled to a peak of 0.95 before it is
analysed (clean clips and mixtures alike). YAMNet reports one set of scores and one
1024-number embedding for each 0.96 s patch, with a new patch every 0.48 s.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import time

import numpy as np
import soundfile as sf
import tensorflow_hub as hub
from scipy.signal import resample_poly

PATCH, HOPS = 0.96, 0.48


def trim_bounds(y):
    n = len(y) // 441
    w = np.abs(y[: n * 441]).reshape(n, 441).max(axis=1)
    nz = np.nonzero(w > 0)[0]
    if len(nz) == 0:
        return 0, 441
    return int(nz[0] * 441), int((nz[-1] + 1) * 441)


def analyse(model, w16):
    w16 = (w16 / (np.max(np.abs(w16)) + 1e-9) * 0.95).astype("float32")
    scores, emb, _ = model(w16)
    return scores.numpy().astype("float16"), emb.numpy().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    t0 = time.time()
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    names = [r["display_name"] for r in csv.DictReader(open(model.class_map_path().numpy().decode(), newline="", encoding="utf-8"))]
    np.save(os.path.join(a.results, "yamnet_class_names.npy"), np.array(names))

    with open(os.path.join(a.data, "meta", "esc50.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sc_all, em_all, owner, ctime, valid, lo_s, hi_s = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        y, sr = sf.read(os.path.join(a.data, "audio", r["filename"]), dtype="float32")
        lo, hi = trim_bounds(y)
        lo_t, hi_t = lo / 44100.0, hi / 44100.0
        w = resample_poly(y, 160, 441).astype("float32")
        sc, em = analyse(model, w)
        starts = np.arange(len(sc)) * HOPS
        overlap = np.minimum(starts + PATCH, hi_t) - np.maximum(starts, lo_t)
        sc_all.append(sc); em_all.append(em); owner.append(np.full(len(sc), i))
        ctime.append(starts + HOPS); valid.append(overlap >= min(0.25, hi_t - lo_t))
        lo_s.append(lo_t); hi_s.append(hi_t)
        if (i + 1) % 400 == 0:
            print(f"[{time.time() - t0:.0f}s] clean {i + 1}/{len(rows)}", flush=True)
    np.savez(os.path.join(a.results, "yam_clean.npz"), scores=np.concatenate(sc_all), emb=np.concatenate(em_all),
             owner=np.concatenate(owner), ctime=np.concatenate(ctime), valid=np.concatenate(valid),
             lo_s=np.array(lo_s), hi_s=np.array(hi_s))

    for tag in ("aug", "test"):
        audio = np.load(os.path.join(a.results, f"mix_{tag}_16k.npy"))
        sc_all, em_all, owner, ctime = [], [], [], []
        for i in range(len(audio)):
            sc, em = analyse(model, audio[i])
            sc_all.append(sc); em_all.append(em); owner.append(np.full(len(sc), i))
            ctime.append(np.arange(len(sc)) * HOPS + HOPS)
        np.savez(os.path.join(a.results, f"yam_{tag}.npz"), scores=np.concatenate(sc_all), emb=np.concatenate(em_all),
                 owner=np.concatenate(owner), ctime=np.concatenate(ctime))
        print(f"[{time.time() - t0:.0f}s] {tag} mixtures done ({len(audio)})", flush=True)
    print("all done")


if __name__ == "__main__":
    main()

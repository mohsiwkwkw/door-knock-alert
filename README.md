# Door-knock alert (ESC-50)

Detect a door knock for a person with hearing loss, and test the everyday sounds that cause false alerts. This is a prototype and not a safety device.

## Reproduce it

1. Download ESC-50 from https://github.com/karolpiczak/ESC-50 (the `master` zip) and extract it so that `data/esc50_raw/ESC-50-master/audio/` exists. The dataset is not stored in this repository (Creative Commons Attribution Non-Commercial).
2. Make a virtual environment and install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (Gradio is also needed for the demo page: `.venv/bin/pip install gradio`).
3. Run in this order from this folder, each as `.venv/bin/python -I src/<script>.py ...` with the arguments shown in the script header: `inspect_data`, `build_features`, `baseline_models`, `run_stress_test`, `bootstrap_contrasts`, then `build_demo_model`, `fold5_report`, `demo_app`.
   `demo/models.joblib` is a Python pickle written by `build_demo_model.py`. Only load pickles from sources you trust; if unsure, delete it and rebuild it yourself.
4. The YAMNet comparison needs a second environment with TensorFlow, TensorFlow Hub and `setuptools<81` (see `requirements-tf-lock.txt`), then `src/yamnet_embed.py` and `src/run_yamnet_eval.py`. First run `src/run_stress_test.py --mixtures-only` to write the 16 kHz mixtures it reads.

## Folder layout

- `data/esc50_raw/` : ESC-50 extracted untouched from `ESC-50-master.zip`. Treat as read-only data. Never run code from inside it.
- `src/` : our scripts. Run them from this folder with `python3 -I src/<script>.py` so nothing in the data folder is imported.
- `results/` : reports and tables written by the scripts.

## The four steps

1. **Clean.** Done for inspection. `src/inspect_data.py` checks the metadata and audio, and measures padding and loudness. The cleaning rule for modeling is below and is applied in the modeling script.
2. **Build, or use or modify a pre-built model.** Planned order: (a) loudness alarm, (b) sound-pattern features (MFCC) with a scikit-learn classifier, (c) a pre-built sound model (YAMNet) used as a feature extractor or as is, (d) the two-wave decision on top.
3. **Train.** Leave-one-fold-out over ESC-50's 5 folds. Never train on test folds, and never train on our own recordings.
4. **Test.** Clean clips, then mixed clips (a knock mixed with background at several volumes), then our own door recordings. Report knocks missed and false alerts per hour, separately.

## Findings from step 1 (real data, 2026-11-14)

- 2,000 clips, 5 folds of 400, 50 classes of 40, 44.1 kHz mono, all files present, no format problems.
- The knock class is `door_wood_knock` (class 30): 40 clips, 8 per fold, from 36 distinct source recordings.
- Four source recordings appear in two folds, but all are non-knock classes (clock tick and alarm, hen and rooster). They do not affect knock detection.
- **Padding shortcut.** Many clips are padded with digital silence. Knock clips average 38% silence (4th highest of 50 classes). 27 of 40 knock clips have padding, against 413 of 1,960 others. Padding fraction alone separates knocks from the rest with AUC 0.75, so a model could cheat by detecting silence. Cleaning rule: do not feed padded regions to the model. Compute features only on the non-silent part, and fill padding with low-level noise when building mixed test clips.
- **Loudness alone is weak.** On raw ESC-50, RMS loudness gives AUC 0.45 (below chance) and peak level 0.62. With the best threshold chosen on training folds, recall is 0.50, about 30% of non-knock clips are flagged, and precision is 3%, while "always say no" is 98% accurate. Quiet-knock tests must therefore rescale volume on purpose.

## Results so far (first pass, clean clips only)

Setup: knock versus everything else, leave-one-fold-out over ESC-50's 5 folds, padding trimmed and volume removed, 132 length-independent sound features (MFCC and spectral summaries). Thresholds are chosen on training folds only. Error bars are 95% Wilson intervals; with only 40 knocks they are wide.

| Detector | Knocks caught (of 40) | Non-knock clips flagged (of 1,960) | Precision |
|---|---|---|---|
| Loudness alarm | 20 (35% to 65%) | 595 (about 30%) | 3% |
| SVM, balanced point | 29 (57% to 84%) | 10 (0.3% to 0.9%) | 74% |
| SVM, high-recall point | 38 (84% to 99%) | 37 (1.4% to 2.6%) | 51% |

- Logistic regression and random forest were slightly worse (average precision 0.78 and 0.73 against 0.86 for the SVM).
- Sounds most often mistaken for knocks: door creaks (4 of 40 flagged), footsteps, drinking and sipping. The random forest also flagged 11 of 40 fireworks clips.
- **Leftover shortcut:** after trimming, clip length alone still separates knocks (AUC 0.73 with the direction flipped; knock clips are shorter). The models do not see length directly, but features such as the maximum and spread over frames might leak it. Next: fixed-length windows around each onset.
- These are clean-clip numbers. The mixing stress test, the two-wave decision and our own recordings are still to do.

## Stress-test results (fixed 0.5 s windows, mixing, two-wave rule)

Script: `src/run_stress_test.py` (about 3 minutes), plots in `results/clean_curves.png` and `results/mixture_recall.png`, error bars from `src/bootstrap_contrasts.py`.

How it works: every detector scans a clip with 0.5 s windows about every 0.1 s, so clip length cannot leak. The SVM is trained leave-one-fold-out in two ways: **clean-trained**, and **mixed-sound-trained** (knocks mixed into home sounds from the training folds at random volumes). **Single-stage** alerts when one window scores above a threshold. **Two-wave** also alerts when two weaker peaks (the threshold minus 0.5, 1.0 or 1.5) fall 0.2 to 2 s apart. Stress test: held-out knocks mixed into held-out home sounds (24 household sound types) at +10 to -10 dB; a hit counts only if an alert falls inside the knock's time span. Operating points are set at a matched false-alert rate on pooled held-out clean negatives, the same way for every detector.

Knocks caught at about 5 false alerts per hour (of ESC-50 non-knock sound):

| Detector | Clean clips | Mixed, knock +10 dB | 0 dB | -10 dB (quiet knock) |
|---|---|---|---|---|
| Single-stage, clean-trained | 72% | 30% | 19% | 11% |
| Two-wave, clean-trained | 75% | 24% | 16% | 10% |
| Single-stage, mixed-sound-trained | 65% | 47% | 35% | 25% |
| Two-wave, mixed-sound-trained | 65% | 49% | 38% | 27% |
| Loudness alarm | 5% (at about 19 false alerts/h) | not tested | | |
| Rhythm rule, no AI (two loudness peaks within 1.5 s) | 3% (cannot get below 28 false alerts/h) | 2% or less | | |

At about 20 false alerts per hour the mixed-sound-trained single-stage model catches 82% of clean clips and 68%, 55% and 41% of mixed ones (+10, 0, -10 dB).

What the 95% intervals say (resampling the 40 knock clips, because they are reused across volumes):
- **Training with mixed sounds helps clearly:** +0.17 more knocks caught across volumes at 5 false alerts per hour (interval +0.12 to +0.22), and +0.24 at 20 per hour. It costs a little on clean clips (72% to 65%).
- **The two-wave rule does not help reliably:** +0.015 on the mixed-sound model at 5 per hour (interval -0.002 to +0.033, could be chance), -0.024 on the clean-trained model, and -0.045 (clearly worse) on the mixed-sound model at 20 per hour. On clean clips the curves are on top of each other at matched false alerts.
- **Both non-AI baselines fail:** they need hundreds to thousands of false alerts per hour to catch most knocks.
- **A quiet knock under household noise is still hard:** even the best detector catches about a quarter (5 per hour) to two in five (20 per hour) at -10 dB.

Caveats, all real:
- Only 40 knocks. ESC-50 knock clips almost all contain several knocks, and 0.5 s windows may already contain a pair, which may be why the two-wave rule adds nothing. A single-knock test needs our own recordings.
- ESC-50 non-knock clips are dense with sound events. "False alerts per hour" is per hour of that sound (2.5 hours in total, 1.2 hours of home sounds; 5 per hour is about 10 alerts), not per hour of a quiet home. Real hours of home noise need our own recordings.
- Thresholds were set on the pooled held-out negatives, so numbers are somewhat optimistic, equally for all detectors. Mixtures are synthetic, not real rooms.
- An earlier run with coarse threshold steps wrongly made the two-wave rule look much better on clean clips. Finer steps removed it. Check curves, not single points.

## Pre-built model: YAMNet (Google's sound classifier, 521 sound types)

Scripts: `src/yamnet_embed.py` (TensorFlow environment) and `src/run_yamnet_eval.py` (main environment, about 5 minutes). Same stress test, same matched false-alert rates. YAMNet scores a 0.96 s patch every 0.48 s, and every waveform is scaled to the same peak first because YAMNet is not gain-invariant.

Three ways to use it: **zero-shot** (its own "Knock" class score, or the larger of "Knock" and "Door"), and **modified** (a logistic regression on its 1024-number embeddings, trained leave-one-fold-out, clean-trained and mixed-sound-trained). Knocks caught at about 5 false alerts per hour:

| Detector | Clean clips | Mixed, knock +10 dB | 0 dB | -10 dB |
|---|---|---|---|---|
| Our SVM, clean-trained | 72% | 30% | 19% | 11% |
| Our SVM, mixed-sound-trained | 65% | 47% | 35% | 25% |
| YAMNet zero-shot "Knock" | 68% | 33% | 25% | 14% |
| YAMNet zero-shot "Knock or Door" | 55% | 24% | 20% | 8% |
| YAMNet + logistic regression, clean-trained | 60% | 20% | 12% | 5% |
| YAMNet + logistic regression, mixed-sound-trained | 57% | 23% | 14% | 6% |

At about 20 false alerts per hour: zero-shot "Knock" catches 85% of clean clips and 50%, 34% and 25% of mixed ones (+10, 0, -10 dB); our mixed-sound-trained SVM catches 82% and 68%, 55% and 41%.

What the 95% intervals say (resampling the 40 knock clips), difference in knocks caught across volumes:
- Zero-shot YAMNet "Knock" minus our mixed-sound-trained SVM: -0.125 at 5 per hour (interval -0.20 to -0.04) and -0.19 at 20 per hour (-0.26 to -0.12). Ours is clearly better under overlapping sounds, but it was trained with exactly that stress and YAMNet was not.
- YAMNet + logistic regression minus our mixed-sound-trained SVM: -0.22 at 5 per hour (-0.29 to -0.14) and -0.24 at 20 per hour (-0.32 to -0.16).

Reading it honestly:
- **A free pre-built model is a strong baseline.** With no training at all, YAMNet's "Knock" class matches our clean-trained SVM on clean clips (68% against 72% at about 5 false alerts per hour; these intervals overlap heavily with only 40 knocks) and does somewhat better than it under mixing.
- **Modifying it by training on its embeddings did not help here** and was worse than zero-shot. Likely reasons, not tested: weak labels (every valid frame of a knock clip counts as a knock, although some frames hold only silence between knocks), 0.96 s patches that are long next to a knock, and a plain logistic regression. This is not proof that the embeddings cannot work.
- **The best result so far is our own windowed SVM trained with mixed sounds.** It does not beat the pre-built model on clean clips, only when sounds overlap.
- Not tested: the phone's built-in door-knock alert (a black box that needs played recordings), and the two-wave rule on YAMNet (its 0.48 s steps are too coarse to test knocks 0.2 s apart).

## Fold-5 scorecard (a simple way to compare any model on the same 400 clips)

Test set: ESC-50 fold 5, never used for training: 400 clips, 8 knocks and 392 other sounds. A knock clip counts as detected if the model raises at least one alert in it; a non-knock clip counts as a false alert if it raises at least one. Script: `src/fold5_report.py`, output `results/fold5_report.json`.

| Measure | Always say "no" | Ours, clean-trained | Ours, mixed-sound-trained |
|---|---|---|---|
| Overall accuracy | 98.0% | 99.25% | 99.0% |
| Knocks detected | 0/8 | 5/8 (62.5%) | 5/8 (62.5%) |
| Knocks missed | 8/8 | 3/8 | 3/8 |
| False alerts | 0/392 | 0/392 (0%) | 1/392 (0.26%) |
| Knock precision | n/a | 100% | 83.3% |
| Knock F1 | 0% | 76.9% | 71.4% |

- Both models miss the same three knock clips (5-218980-A-30, 5-218981-A-30, 5-250026-A-30). The mixed-sound model's single false alert is a drinking sound.
- With 8 knocks, one clip is 12.5 points and the 95% interval for "knocks caught" is 31% to 86%, so a one-knock difference between models means little. False alerts and F1 separate models better here, but 392 clips is still small. The more reliable test is all five folds (40 knocks): the first-pass whole-clip SVM caught 29/40 with 10 false-alert clips out of 1,960 (precision 74%, F1 73%).
- Compare other models only on this exact 400-clip fold, with the same counting rule.

## Demo page and write-up

- **Demo page:** `src/demo_app.py` (Gradio). Start it from this folder with `.venv/bin/python -I src/demo_app.py` and open http://127.0.0.1:7860. Tabs: try a clip (all 8 held-out knock clips and six household sounds as examples, or upload or record your own), a volume stress test (mix a knock into a household sound and move the volume), and results and limits. Models come from `src/build_demo_model.py` (trained on folds 1 to 4 only; every example clip is from fold 5, so never seen in training). Checked in the browser: both tabs run end to end.
- **What the demo honestly shows:** on the 8 held-out knock clips each model catches 5 and misses 3. On 392 held-out non-knock clips (0.5 hours) the clean-trained model flagged none of them and the mixed-sound model flagged 1 clip (a drinking sound, with 2 alerts). See the fold-5 scorecard above. For the default knock mixed into the first clip of each of 24 household sounds, the mixed-sound model catches it in 17 at +10 dB, 12 at 0 dB and 9 at -10 dB; a vacuum cleaner or washing machine defeats it at every volume.
- **The loudness alarm in the demo** has a level you set yourself (default -25 dB), only to show how it reacts. It is not the tuned baseline from the experiments, which caught 5% of knocks at about 19 false alerts per hour.
- **Thresholds** for the demo models come from the pooled experiment, so a single retrained model's false-alert rate can differ a little.
- **Sharing the demo:** add `--share` (`.venv/bin/python -I -u src/demo_app.py --share`) for a temporary public `gradio.live` link, good for up to a week while the script keeps running on this computer. The `-u` makes the link appear in the log. The server may only hand out the example clips and two chart images (checked: other project files return 403 locally and through the public link). The first use makes Gradio download a small tunnel helper (`frpc`, about 12 MB) into `~/.cache/huggingface/gradio/frpc/`. Stop it with `kill $(ss -ltnp | grep ':7860' | sed -E 's/.*pid=([0-9]+).*/\1/' | head -1)`. Without `--share` the demo stays local-only.
- **Audio formats:** the upload box accepts WAV, MP3, FLAC, OGG and also M4A (iPhone voice memos). M4A and anything else the Python audio reader cannot open is decoded by GStreamer, which is already installed on this machine (no download): the upload is copied to a plain temporary name, no shell is used, files over 25 MB are refused, and the job times out after 40 seconds. Checked end to end through the real upload path: an M4A recording works, a normal WAV still works, and a text file renamed `.m4a` gives a clear error message. This relies on GStreamer's AAC decoder being installed, which it is here.
- **Write-up draft:** `WRITEUP.md`, following the hackathon's template, with real numbers and clearly marked placeholders for team names, links and the real-recording results.

## Scripts

- `src/inspect_data.py`: step 1 data check. Run, results in `results/`.
- `src/build_features.py`: applies the cleaning rule and builds features. Run: 2,000 clips, 132 features each, about 4 minutes, output `results/features.npz`.
- `src/build_demo_model.py`: trains the two demo models on folds 1 to 4 and saves them in `demo/`. Run, about 3 minutes.
- `src/demo_app.py`: the Gradio demo page. Gradio 6.32 is installed in the main environment (added without changing any existing package).
- `src/yamnet_embed.py`: extracts YAMNet outputs for every clip and mixture (TensorFlow environment). Run, about 1 minute.
- `src/run_yamnet_eval.py`: evaluates zero-shot and modified YAMNet against our SVM. Run, about 5 minutes.
- `src/run_stress_test.py`: fixed-window detector, two-wave rule, mixing stress test. Run, about 3 minutes.
- `src/bootstrap_contrasts.py`: error bars for the key comparisons. Run.
- `src/baseline_models.py`: leave-one-fold-out training and testing of three models. Run: about 30 seconds, outputs `results/baselines.json` and `results/clip_scores.csv`.

## Environment

The machine's own Python (3.12) has no pip or venv package, so pip was bootstrapped inside a project-only virtual environment `.venv/` with the official `get-pip.py` (nothing in the system was changed). Packages come from `requirements.txt`; exact versions are in `requirements-lock.txt`. Run scripts with `.venv/bin/python -I src/<script>.py` from this folder. Do not commit `.venv/`. No GPU; everything planned here runs on CPU. TensorFlow 2.21 and TensorFlow Hub live in a second environment `.venv-tf/` (versions in `requirements-tf-lock.txt`), kept separate so they cannot break the main one; run its scripts with `TFHUB_CACHE_DIR=$PWD/cache/tfhub .venv-tf/bin/python -I src/<script>.py`. TensorFlow Hub needs `setuptools<81`, so it is pinned there. YAMNet itself (18 MB) is cached in `cache/tfhub/`. The large files in `results/` (the 16 kHz mixtures and YAMNet outputs) can be regenerated with the scripts and deleted to save disk.

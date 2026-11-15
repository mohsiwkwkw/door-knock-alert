# Did someone knock? A door-knock alert, stress-tested against household noise

**Track:** Health
**Team:** [TEAM TO FILL: names, schools]

*This project is a prototype and is not medical advice. It is not a safety device.*

## The problem
A person with hearing loss can miss a visitor at the door. A plain loudness alarm fires on any loud sound, so it is both annoying and unreliable. Phones already have AI door-knock alerts (Apple's Sound Recognition, Android's Sound Notifications), but how such detectors behave when a knock overlaps a vacuum cleaner or a running tap is rarely measured in the open.

## What we built
A detector that scans sound in 0.5 second windows and flags a possible knock, plus a small demo page where you play a held-out clip or mix a real knock into a household sound and move the volume. [TEAM TO FILL: link to the demo or video.]

## What we added
We started from one public dataset, then added our own question: **what happens when household sounds overlap the knock, and does it help to train for that, or to require two knocks?** We built a mixing stress test (knocks mixed into 24 types of household sound, 10 dB louder to 10 dB quieter), compared against non-AI rules and Google's pre-built YAMNet model, and tested a "people knock twice" rule. [TEAM TO FILL: results on our own recordings of real doors and household noise.]

## How it works
**Data:** ESC-50, 2,000 five-second clips, 40 of them door knocks, in 5 predefined folds. We never train on the fold we test on.
**Cleaning:** many clips are padded with digital silence, and padding alone told knocks from other sounds (AUC 0.75), so a model could cheat. We trim the padding, remove overall volume, and use fixed-length windows.
**Model:** an SVM on sound-pattern features (MFCC and spectral summaries), trained two ways: on clean clips, and with knocks mixed into home sounds at random volumes.
**Fair comparison:** every detector is compared at the same false-alert rate (about 5 and 20 per hour of ESC-50 non-knock sound).

## Results
Knocks caught at about 5 false alerts per hour:

| Detector | Clean clips | Knock 10 dB louder | Same volume | 10 dB quieter |
|---|---|---|---|---|
| Loudness alarm (baseline) | 5% (at about 19 alerts/h) | | | |
| Rule: two loudness peaks (no AI) | 3% (cannot get below 28 alerts/h) | | | |
| Our model, clean-trained | 72% | 30% | 19% | 11% |
| Our model, trained with mixed sounds | 65% | 47% | 35% | 25% |
| Google YAMNet "Knock", no training | 68% | 33% | 25% | 14% |

- Training with mixed sounds helped clearly: +0.17 more knocks caught across volumes (95% interval +0.12 to +0.22).
- The "knock twice" rule did not help reliably (+0.015, interval -0.002 to +0.033). Modifying YAMNet with our own classifier did worse than using it as is.
- **Where it fails:** a quiet knock under a vacuum cleaner or washing machine is missed; it confuses door creaks, footsteps and sipping with knocks; and on held-out clips both models still miss 3 of 8 knocks. With only 40 knocks the intervals are wide, ESC-50 false alerts are per hour of ESC-50 sound and not of a quiet home, and our mixtures are synthetic.

## What we'd do next
1. Test on our own recordings of real doors (single knocks, spacing) and hours of household noise.
2. Compare against the phone's built-in alert on the same recordings.
3. Label YAMNet training frames properly, which we believe is why our attempt to modify it failed.

## Sources and credits
- ESC-50 dataset, K. Piczak, 2015 (Creative Commons Attribution Non-Commercial; clips from Freesound): https://github.com/karolpiczak/ESC-50
- YAMNet, Google, trained on AudioSet, via TensorFlow Hub: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- Libraries: librosa, scikit-learn, scipy, numpy, matplotlib, soundfile, TensorFlow and TensorFlow Hub, Gradio.
- AI tools: Claude (Anthropic, via Claude Code) helped plan the experiments and write and run the code; Gemini helped generate the first list of project ideas. We reviewed and can explain all of it. [TEAM: confirm everyone has read and understood the code.]
- Rules and template: Veritas AI International High School Hackathon 2026 project briefs.

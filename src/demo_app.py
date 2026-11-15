"""Demo page: "Did someone knock?" (prototype, not a safety device).

Three tabs: try a clip, a volume stress test (mix a knock into a home sound), and results.
The two models were trained on ESC-50 folds 1 to 4 only (see build_demo_model.py); every
example clip comes from fold 5, so the models have never seen them.

Run from the doorknock folder:
    .venv/bin/python -I src/demo_app.py
then open http://127.0.0.1:7860
Add --share to get a temporary public link (gradio.live) that works for anyone you send it to
while this script keeps running. The server may only hand out the example clips and two chart
images, not the rest of the project.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_stress_test as rst  # noqa: E402  (feature code shared with the experiments)

import gradio as gr  # noqa: E402
import joblib  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DATA = os.path.join(ROOT, "data", "esc50_raw", "ESC-50-master")
AUDIO = os.path.join(DATA, "audio")
MODELS = joblib.load(os.path.join(ROOT, "demo", "models.joblib"))
CFG = json.load(open(os.path.join(ROOT, "demo", "config.json"), encoding="utf-8"))
SR = rst.SR
DISCLAIMER = ("**Prototype, not a safety device or medical advice.** Trained on a small public dataset "
              "(ESC-50, 40 knock clips). It will miss knocks and raise false alerts.")


MAX_BYTES = 25_000_000


def _decode_with_gstreamer(path):
    """Turn formats libsndfile cannot read (such as iPhone .m4a) into a 44.1 kHz mono WAV.

    Uses GStreamer, which is already installed on this machine (no download). The upload is
    copied to a plain temporary name first so odd file names cannot reach the command, no shell
    is involved, and the job is size- and time-limited.
    """
    if os.path.getsize(path) > MAX_BYTES:
        raise gr.Error("That file is too large (limit 25 MB).")
    tmp = tempfile.mkdtemp(prefix="knock_")
    src, dst = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.wav")
    shutil.copyfile(path, src)
    cmd = ["gst-launch-1.0", "-q", "filesrc", f"location={src}", "!", "decodebin", "!", "audioconvert", "!",
           "audioresample", "!", f"audio/x-raw,format=S16LE,rate={SR},channels=1", "!", "wavenc", "!",
           "filesink", f"location={dst}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=40)
    except (subprocess.SubprocessError, OSError) as err:
        shutil.rmtree(tmp, ignore_errors=True)
        raise gr.Error("Could not read that audio file. Try WAV, MP3 or M4A.") from err
    return dst, tmp


def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=30)
    except Exception:
        wav, tmp = _decode_with_gstreamer(path)
        try:
            y, _ = librosa.load(wav, sr=SR, mono=True, duration=30)
        except Exception as err:
            raise gr.Error("Could not read that audio file. Try WAV, MP3 or M4A.") from err
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    y = y.astype("float32")
    if len(y) < SR // 4:
        raise gr.Error("That clip is too short (under a quarter of a second).")
    return y


def analyse(y, loud_level):
    lo, hi = rst.trim_bounds(y)
    F, r, _ = rst.frame_feats(y)
    T = F.shape[0]
    a0, b0 = lo // rst.HOP, min(T, -(-hi // rst.HOP))
    X, t = rst.windows_of(F, r, a0, b0)
    res = {}
    for key, thr in (("clean", CFG["threshold_clean"]), ("aug", CFG["threshold_aug"])):
        s = MODELS[key].decision_function(X)
        res[key] = dict(scores=s, t=t, thr=thr, alerts=rst.alerts_model(s, t, thr))
    res["loud"] = dict(alerts=rst.alerts_loud(r, a0, b0, loud_level), r=r, level=loud_level)
    return res


def say(name, alerts, span=None):
    if not alerts:
        return f"- **{name}:** no alert"
    when = ", ".join(f"{a:.1f} s" for a in alerts[:4])
    if span is None:
        return f"- **{name}:** ALERT, possible door knock at {when}"
    inside = [a for a in alerts if span[0] - 0.2 <= a <= span[1] + 1.0]
    outside = [a for a in alerts if a not in inside]
    parts = ["caught the knock" if inside else "MISSED the knock"]
    if outside:
        parts.append(f"false alert at {', '.join(f'{a:.1f} s' for a in outside[:3])}")
    return f"- **{name}:** " + "; ".join(parts)


def draw(res, span=None):
    fig = Figure(figsize=(8, 5.2))
    ax, ax2 = fig.subplots(2, 1, sharex=True)
    for key, label, color in (("clean", "model trained on clean clips", "#d95f02"),
                              ("aug", "model trained with mixed sounds", "#1b9e77")):
        d = res[key]
        ax.plot(d["t"], d["scores"], label=label, color=color)
        ax.axhline(d["thr"], color=color, ls="--", lw=0.8)
        for a in d["alerts"]:
            ax.axvline(a, color=color, ls=":", lw=1.2)
    r = res["loud"]["r"]
    times = np.arange(len(r)) * rst.SEC
    ax2.plot(times, np.where(r < -100, np.nan, r), color="#7570b3", label="loudness (dB)")
    ax2.set_ylim(-95, 0)
    ax2.axhline(res["loud"]["level"], color="#7570b3", ls="--", lw=0.8)
    for a in res["loud"]["alerts"]:
        ax2.axvline(a, color="#7570b3", ls=":", lw=1.2)
    if span:
        for axx in (ax, ax2):
            axx.axvspan(span[0], span[1], color="grey", alpha=0.15)
    ax.set_ylabel("knock score (dashed = alert level)")
    ax2.set_ylabel("loudness (dB)")
    ax2.set_xlabel("seconds (grey band = where the knock really is)" if span else "seconds")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def try_clip(path, loud_level):
    if not path:
        return "Pick an example or upload a clip first.", None
    y = load_audio(path)
    res = analyse(y, loud_level)
    text = "\n".join([say("Loudness alarm (level set by you)", res["loud"]["alerts"]),
                      say("Our model, trained on clean clips", res["clean"]["alerts"]),
                      say("Our model, trained with mixed sounds", res["aug"]["alerts"])])
    return text + "\n\n" + DISCLAIMER, draw(res)


def stress(knock_name, bg_cat, bg_num, snr, loud_level):
    k, _ = sf.read(os.path.join(AUDIO, knock_name), dtype="float32")
    names = CFG["demo_home"][bg_cat]
    bg, _ = sf.read(os.path.join(AUDIO, names[int(bg_num) % len(names)]), dtype="float32")
    kl, kh = rst.trim_bounds(k)
    bl, bh = rst.trim_bounds(bg)
    mix, off, L = rst.make_mix(k, kl, kh, bg, bl, bh, float(snr), np.random.default_rng(1))
    res = analyse(mix, loud_level)
    span = (off / SR, (off + L) / SR)
    text = "\n".join([say("Loudness alarm (level set by you)", res["loud"]["alerts"], span),
                      say("Our model, trained on clean clips", res["clean"]["alerts"], span),
                      say("Our model, trained with mixed sounds", res["aug"]["alerts"], span)])
    play = (mix / (np.max(np.abs(mix)) + 1e-9) * 0.9).astype("float32")
    return (SR, play), text + "\n\n" + DISCLAIMER, draw(res, span)


ABOUT = f"""
### What this is
A prototype that listens for door knocks for a person with hearing loss and tests the everyday sounds most likely to trigger a nuisance alert. All numbers come from the public ESC-50 dataset (2,000 clips, 40 of them knocks), tested on folds the models never trained on.

### Results (about 5 false alerts per hour of ESC-50 non-knock sound; knocks caught)
| Detector | Clean clips | Knock 10 dB louder than home sound | Same volume | 10 dB quieter |
|---|---|---|---|---|
| Loudness alarm | 5% (at about 19 false alerts/h) | not tested | | |
| Our model, trained on clean clips | 72% | 30% | 19% | 11% |
| Our model, trained with mixed sounds | 65% | 47% | 35% | 25% |
| Google YAMNet "Knock" class, no training | 68% | 33% | 25% | 14% |

- Training with mixed sounds is a clear gain (+0.17 more knocks caught across volumes, 95% interval +0.12 to +0.22).
- A quiet knock under household noise is still hard: even the best model catches about a quarter.
- The "people knock twice" rule did not help reliably in this test.
- Only 40 knocks, no real doors yet, and ESC-50 false alerts are per hour of ESC-50 sound, not of a quiet home. Real recordings are the next test.

{DISCLAIMER}
"""


EX_NEG = ("footsteps", "clock_tick", "door_wood_creaks", "clapping", "can_opening", "keyboard_typing")


def example_files():
    """The only files the web server may hand out: the example clips and two chart images."""
    files = [os.path.join(AUDIO, n) for n in CFG["demo_knocks"]]
    files += [os.path.join(AUDIO, CFG["demo_home"][c][0]) for c in EX_NEG]
    files += [os.path.join(ROOT, "results", i) for i in ("mixture_recall.png", "clean_curves.png")]
    return files


def build_ui():
    knocks = CFG["demo_knocks"]
    homes = sorted(CFG["demo_home"])
    examples = [[os.path.join(AUDIO, n), -25] for n in knocks]
    for c in EX_NEG:
        examples.append([os.path.join(AUDIO, CFG["demo_home"][c][0]), -25])
    with gr.Blocks(title="Did someone knock?") as demo:
        gr.Markdown("# Did someone knock?\nA door-knock alert prototype for a person with hearing loss. " + DISCLAIMER)
        with gr.Tab("Try a clip"):
            gr.Markdown("Pick an example (clips the models have never seen), or upload or record your own. "
                        "The first 8 examples are every held-out knock clip, not a chosen selection: on these 8 each model "
                        "catches 5 and misses 3. On the 392 held-out non-knock clips (half an hour of sound) the clean-trained "
                        "model flagged none of them and the mixed-sound model flagged 1 (a drinking sound).")
            audio = gr.Audio(type="filepath", sources=["upload", "microphone"],
                             label="Sound clip (WAV, MP3, M4A and similar; iPhone voice memos work)")
            level = gr.Slider(-50, -3, value=-25, step=1, label="Loudness alarm level (dB). Louder than this raises an alert. Set by you, not tuned.")
            btn = gr.Button("Listen for a knock", variant="primary")
            out_text = gr.Markdown()
            out_plot = gr.Plot()
            gr.Examples(examples=examples, inputs=[audio, level], label="Example clips (never seen in training)")
            btn.click(try_clip, [audio, level], [out_text, out_plot])
        with gr.Tab("Volume stress test"):
            gr.Markdown("Mix a real knock into a household sound and move the volume. Negative dB means the knock is quieter than the background. "
                        "The knock selected at first is one the models catch on its own; three of the eight are missed even without background. "
                        "For that knock, mixed one at a time into the first clip of each of the 24 household sounds, the mixed-sound model catches it in "
                        "17 of 24 at +10 dB, 12 of 24 at the same volume (0 dB) and 9 of 24 at -10 dB. A vacuum cleaner or washing machine defeats it at every volume. "
                        "The starting settings below (typing, +10 dB) are one of the cases it catches.")
            k = gr.Dropdown(knocks, value=knocks[2], label="Knock clip")
            c = gr.Dropdown(homes, value="keyboard_typing" if "keyboard_typing" in homes else homes[0], label="Household sound")
            n = gr.Slider(1, 8, value=1, step=1, label="Which clip of that sound (1 to 8)")
            s = gr.Slider(-12, 12, value=10, step=1, label="Knock volume relative to the household sound (dB)")
            lv = gr.Slider(-50, -3, value=-25, step=1, label="Loudness alarm level (dB)")
            b2 = gr.Button("Mix and listen", variant="primary")
            mixed = gr.Audio(label="The mixed sound", type="numpy")
            t2 = gr.Markdown()
            p2 = gr.Plot()
            b2.click(stress, [k, c, n, s, lv], [mixed, t2, p2])
        with gr.Tab("Results and limits"):
            gr.Markdown(ABOUT)
            for img in ("mixture_recall.png", "clean_curves.png"):
                path = os.path.join(ROOT, "results", img)
                if os.path.exists(path):
                    gr.Image(path, label=img, show_label=True)
    return demo


if __name__ == "__main__":
    # Local only by default. Add --share for a temporary public link (through Gradio's relay).
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 7860
    build_ui().launch(server_name="127.0.0.1", server_port=port, show_error=True,
                      allowed_paths=example_files(), share="--share" in sys.argv)

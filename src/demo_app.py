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

Batches in this file
   1  Tools, paths, and loading the saved models
   2  Read an audio file (including iPhone .m4a)
   3  The detector: one waveform in, scores and alert times out
   4  Say the result in words and draw the chart
   5  The two button actions (try a clip, volume stress test)
   6  Text for the Results tab
   7  Which files the web server may share
   8  The look of the page (colours, fonts, header)
   9  The page layout: three tabs and their buttons
  10  Start the server
"""
# ===== BATCH 1 | Tools, paths, and loading the saved models =====
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


# ===== BATCH 2 | Read an audio file (including iPhone .m4a) =====
MAX_BYTES = 25_000_000     # biggest upload we accept (25 MB)


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
    """Load any common audio file as 44.1 kHz mono, at most 30 seconds; fall back to GStreamer for .m4a."""
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


# ===== BATCH 3 | The detector: one waveform in, scores and alert times out =====
def analyse(y, loud_level):
    """Run the loudness alarm and both models on one waveform. Uses the same functions as the experiments."""
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


# ===== BATCH 4 | Say the result in words and draw the chart =====
def say(name, alerts, span=None):
    """One bullet of text for one detector. If we know where the knock really is (span), say caught or MISSED."""
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
    """Chart: model scores with alert levels on top, loudness on the bottom."""
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
    for axx in (ax, ax2):  # thin grey frame and light grid to match the page
        axx.grid(color="#eaecef", lw=0.8)
        for sp in axx.spines.values():
            sp.set_color("#d1d5da")
    fig.tight_layout()
    return fig


# ===== BATCH 5 | The two button actions (try a clip, volume stress test) =====
def try_clip(path, loud_level):
    """Button 1: load the chosen clip, run all detectors, return the text and the chart."""
    if not path:
        return "Pick an example or upload a clip first.", None
    y = load_audio(path)
    res = analyse(y, loud_level)
    text = "\n".join([say("Loudness alarm (level set by you)", res["loud"]["alerts"]),
                      say("Our model, trained on clean clips", res["clean"]["alerts"]),
                      say("Our model, trained with mixed sounds", res["aug"]["alerts"])])
    return text + "\n\n" + DISCLAIMER, draw(res)


def stress(knock_name, bg_cat, bg_num, snr, loud_level):
    """Button 2: mix a real knock into a household sound at the chosen volume, then run all detectors."""
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


# ===== BATCH 6 | Text for the Results tab =====
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


# ===== BATCH 7 | Which files the web server may share =====
EX_NEG = ("footsteps", "clock_tick", "door_wood_creaks", "clapping", "can_opening", "keyboard_typing")


def example_files():
    """The only files the web server may hand out: the example clips and two chart images."""
    files = [os.path.join(AUDIO, n) for n in CFG["demo_knocks"]]
    files += [os.path.join(AUDIO, CFG["demo_home"][c][0]) for c in EX_NEG]
    files += [os.path.join(ROOT, "results", i) for i in ("mixture_recall.png", "clean_curves.png")]
    return files


# ===== BATCH 8 | The look of the page (colours, fonts, header) =====
# Look: a plain 2017-style code-hosting page (white page, dark top bar, thin grey borders,
# system fonts, blue links, green button). Style only; it loads no fonts or files from other sites.
GH_FONT = ("-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica", "Arial", "sans-serif")
GH_MONO = ("SFMono-Regular", "Consolas", "Liberation Mono", "Menlo", "Courier", "monospace")
THEME = gr.themes.Base(font=GH_FONT, font_mono=GH_MONO, radius_size=gr.themes.sizes.radius_sm)
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"

HEADER = """
<div class="gh-bar"><span class="gh-brand">Did someone knock?</span><span class="gh-bar-note">door-knock alert prototype</span></div>
<div class="gh-head">
  <h1>Did someone knock?</h1>
  <p>A door-knock alert prototype for a person with hearing loss. Two small models (SVMs) listen to half-second pieces of sound and score each one.</p>
  <span class="gh-lbl gh-y">prototype</span><span class="gh-lbl gh-r">not a safety device</span><span class="gh-lbl gh-b">ESC-50 dataset</span><span class="gh-lbl gh-g">trained on folds 1 to 4</span>
</div>
<div class="gh-flash"><b>Prototype, not a safety device or medical advice.</b> Trained on a small public dataset (ESC-50, 40 knock clips). It will miss knocks and raise false alerts.</div>
"""

GH_CSS = """
:root, .dark, body, .gradio-container {
  --body-background-fill: #ffffff; --background-fill-primary: #ffffff; --background-fill-secondary: #f6f8fa;
  --body-text-color: #24292e; --body-text-color-subdued: #586069; --block-title-text-color: #24292e; --block-label-text-color: #24292e;
  --block-background-fill: #ffffff; --block-border-color: #d1d5da; --border-color-primary: #d1d5da; --block-border-width: 1px;
  --block-radius: 3px; --block-shadow: none; --block-label-background-fill: #f6f8fa; --block-label-border-color: #d1d5da;
  --input-background-fill: #fafbfc; --input-background-fill-focus: #ffffff; --input-border-color: #d1d5da; --input-radius: 3px;
  --input-shadow: inset 0 1px 2px rgba(27,31,35,.075); --input-shadow-focus: inset 0 1px 2px rgba(27,31,35,.075), 0 0 0 3px rgba(3,102,214,.3);
  --input-border-color-focus: #2188ff; --color-accent: #0366d6; --color-accent-soft: #f1f8ff; --link-text-color: #0366d6; --link-text-color-hover: #0366d6;
  --link-text-color-active: #0366d6; --link-text-color-visited: #0366d6; --checkbox-label-background-fill: #fafbfc;
  --table-even-background-fill: #ffffff; --table-odd-background-fill: #f6f8fa; --table-border-color: #dfe2e5; --table-row-focus: #f1f8ff;
  --button-primary-background-fill: linear-gradient(-180deg, #34d058 0%, #28a745 90%); --button-primary-background-fill-hover: linear-gradient(-180deg, #2fcb53 0%, #269f42 90%);
  --button-primary-text-color: #ffffff; --button-primary-border-color: rgba(27,31,35,.2); --button-primary-border-color-hover: rgba(27,31,35,.2);
  --button-secondary-background-fill: linear-gradient(-180deg, #fafbfc 0%, #eff3f6 90%); --button-secondary-background-fill-hover: linear-gradient(-180deg, #f0f3f6 0%, #e6ebf1 90%);
  --button-secondary-text-color: #24292e; --button-secondary-border-color: rgba(27,31,35,.2);
  --slider-color: #0366d6; --shadow-drop: none; --shadow-drop-lg: none;
}
body, .gradio-container { background: #ffffff !important; color: #24292e; font-size: 14px; line-height: 1.5; }
.gradio-container { max-width: 980px !important; width: 100% !important; min-width: 0 !important; margin: 0 auto !important; padding: 0 16px 40px !important; }
@media (max-width: 600px) { .gh-bar { flex-wrap: wrap; row-gap: 2px; } .gh-head h1 { font-size: 18px; } }
footer { display: none !important; }
html, body { overflow-x: hidden !important; }
.gradio-container { overflow: visible !important; }

/* top bar, full width */
.gh-bar { background: #24292e; color: #ffffff; padding: 16px 4px; display: flex; align-items: baseline; gap: 14px; margin: 0 -16px;
  box-shadow: 0 0 0 100vmax #24292e; clip-path: inset(0 -100vmax); }
.gh-bar .gh-brand { font-weight: 600; font-size: 16px; color: #ffffff !important; }
.gh-bar .gh-bar-note { font-size: 12px; color: #c8ccd0 !important; }
/* title area */
.gh-head { background: #fafbfc; margin: 0 -16px; padding: 24px 4px 16px; border-bottom: 1px solid #e1e4e8;
  box-shadow: 0 0 0 100vmax #fafbfc; clip-path: inset(0 -100vmax); }
.gh-head h1 { font-size: 20px; font-weight: 400; color: #0366d6 !important; margin: 0 0 4px; }
.gh-head p { margin: 0 0 10px; color: #586069; }
.gh-lbl { display: inline-block; font-size: 12px; font-weight: 600; line-height: 1; padding: 4px 7px; margin-right: 6px; border-radius: 3px; color: #fff; }
.gh-y { background: #e6a700; color: #24292e; } .gh-r { background: #d73a49; } .gh-b { background: #0366d6; } .gh-g { background: #28a745; }
.gh-flash { margin: 16px -12px 12px; padding: 10px 16px; background: #fffbdd; color: #735c0f; border: 1px solid #d9d0a5; border-radius: 3px; }

/* tabs look like repository tabs */
.tab-wrapper, .tab-container { border-bottom: 1px solid #e1e4e8 !important; }
button[role="tab"] { color: #586069 !important; background: transparent !important; border: 1px solid transparent !important; border-bottom: 0 !important;
  border-radius: 3px 3px 0 0 !important; padding: 8px 16px !important; font-size: 14px !important; font-weight: 400 !important; margin-bottom: -1px; box-shadow: none !important; }
button[role="tab"]:hover { color: #24292e !important; }
button[role="tab"].selected, button[role="tab"][aria-selected="true"] { color: #24292e !important; background: #ffffff !important; border-color: #e1e4e8 !important;
  border-top: 3px solid #f9826c !important; font-weight: 600 !important; }
button[role="tab"].selected::after, .tab-container::after, .tab-wrapper::after { display: none !important; background: transparent !important; }

/* buttons: green and grey, like 2017 */
button.primary, .primary { border-radius: 3px !important; font-size: 14px !important; font-weight: 600 !important; border: 1px solid rgba(27,31,35,.2) !important;
  background-color: #28a745 !important; background-image: linear-gradient(-180deg, #34d058 0%, #28a745 90%) !important; color: #fff !important; box-shadow: none !important; }
button.primary:hover { background-image: linear-gradient(-180deg, #2fcb53 0%, #269f42 90%) !important; }
button.secondary { border-radius: 3px !important; font-weight: 600 !important; border: 1px solid rgba(27,31,35,.2) !important; box-shadow: none !important;
  background-image: linear-gradient(-180deg, #fafbfc 0%, #eff3f6 90%) !important; color: #24292e !important; }

/* panels, labels, inputs */
.block, .form { border-radius: 3px !important; box-shadow: none !important; }
label > span, .block-title, .label-wrap span { font-weight: 600; color: #24292e; font-size: 14px; }
input[type="number"], textarea, input[type="text"], select { border: 1px solid #d1d5da !important; border-radius: 3px !important; background: #fafbfc !important; box-shadow: inset 0 1px 2px rgba(27,31,35,.075) !important; }
input:focus, textarea:focus, select:focus { border-color: #2188ff !important; background: #fff !important; outline: none !important; box-shadow: inset 0 1px 2px rgba(27,31,35,.075), 0 0 0 3px rgba(3,102,214,.3) !important; }
input[type="range"] { accent-color: #0366d6; }

/* text like a rendered readme */
.prose, .md, .markdown { color: #24292e; font-size: 14px; line-height: 1.5; }
.prose a, .md a { color: #0366d6; text-decoration: none; } .prose a:hover, .md a:hover { text-decoration: underline; }
.prose h1, .prose h2, .prose h3, .md h1, .md h2, .md h3 { color: #24292e; font-weight: 600; line-height: 1.25; margin: 20px 0 12px; }
.prose h2, .md h2 { font-size: 21px; padding-bottom: 5px; border-bottom: 1px solid #eaecef; }
.prose h3, .md h3 { font-size: 17px; }
.prose code, .md code { background: rgba(27,31,35,.05); border-radius: 3px; padding: 2px 5px; font-size: 85%; }
.prose table, .md table { border-collapse: collapse; width: 100%; display: table; margin: 0 0 12px; }
.prose th, .prose td, .md th, .md td { border: 1px solid #dfe2e5 !important; padding: 6px 13px; background: transparent; }
.prose th, .md th { font-weight: 600; background: #ffffff; }
.prose tr:nth-child(2n) td, .md tr:nth-child(2n) td { background: #f6f8fa; }
.prose ul, .md ul { padding-left: 2em; }
.block.readme { border: 1px solid #d1d5da !important; border-radius: 3px !important; padding: 0 !important; overflow: hidden; background: #fff; }
.block.readme::before { content: "Results and limits"; display: block; padding: 10px 16px; background: #f6f8fa; border-bottom: 1px solid #d1d5da; font-weight: 600; }
.prose.readme { border: 0 !important; padding: 4px 24px 16px !important; }
.block.flash-out { border-left: 3px solid #28a745 !important; }
"""


# ===== BATCH 9 | The page layout: three tabs and their buttons =====
def build_ui():
    knocks = CFG["demo_knocks"]
    homes = sorted(CFG["demo_home"])
    examples = [[os.path.join(AUDIO, n), -25] for n in knocks]
    for c in EX_NEG:
        examples.append([os.path.join(AUDIO, CFG["demo_home"][c][0]), -25])
    with gr.Blocks(title="Did someone knock?") as demo:
        gr.HTML(HEADER)
        with gr.Tab("Try a clip"):
            gr.Markdown("Pick an example (clips the models have never seen), or upload or record your own. "
                        "The first 8 examples are every held-out knock clip, not a chosen selection: on these 8 each model "
                        "catches 5 and misses 3. On the 392 held-out non-knock clips (half an hour of sound) the clean-trained "
                        "model flagged none of them and the mixed-sound model flagged 1 (a drinking sound).")
            audio = gr.Audio(type="filepath", sources=["upload", "microphone"],
                             label="Sound clip (WAV, MP3, M4A and similar; iPhone voice memos work)")
            level = gr.Slider(-50, -3, value=-25, step=1, label="Loudness alarm level (dB). Louder than this raises an alert. Set by you, not tuned.")
            btn = gr.Button("Listen for a knock", variant="primary")
            out_text = gr.Markdown(elem_classes=["flash-out"])
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
            t2 = gr.Markdown(elem_classes=["flash-out"])
            p2 = gr.Plot()
            b2.click(stress, [k, c, n, s, lv], [mixed, t2, p2])
        with gr.Tab("Results and limits"):
            gr.Markdown(ABOUT, elem_classes=["readme"])
            for img in ("mixture_recall.png", "clean_curves.png"):
                path = os.path.join(ROOT, "results", img)
                if os.path.exists(path):
                    gr.Image(path, label=img, show_label=True)
    return demo


# ===== BATCH 10 | Start the server =====
if __name__ == "__main__":
    # Local only by default. Add --share for a temporary public link (through Gradio's relay).
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 7860
    build_ui().launch(server_name="127.0.0.1", server_port=port, show_error=True,
                      allowed_paths=example_files(), share="--share" in sys.argv,
                      theme=THEME, css=GH_CSS, js=FORCE_LIGHT)

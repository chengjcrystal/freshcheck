"""FreshCheck: fruit freshness tester web app.

A clean, product-style Gradio site. Upload a fruit photo and a small fine-tuned
CNN reports whether it's fresh or spoiled, with a confidence score.
"""

import os

import gradio as gr
from PIL import Image

from inference import FreshnessClassifier, CKPT_PATH

clf = FreshnessClassifier() if os.path.exists(CKPT_PATH) else None

# ---------------------------------------------------------------- styling ----
THEME = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --accent:#059669; }
.gradio-container {
    background: #ffffff !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 20px 40px !important;
    color: var(--ink);
}
footer { display: none !important; }

/* ---- top nav ---- */
#nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 4px; border-bottom: 1px solid var(--line); margin-bottom: 8px;
}
#nav .brand { display:flex; align-items:center; gap:9px; font-weight:800; font-size:1.15rem; color:var(--ink); }
#nav .brand .dot { width:26px; height:26px; border-radius:8px;
    background:linear-gradient(135deg,#10b981,#059669); display:inline-flex;
    align-items:center; justify-content:center; font-size:15px; }
#nav .links { display:flex; gap:26px; font-size:.92rem; font-weight:600; color:var(--muted); }
#nav .links span { cursor:default; }
#nav .cta { background:var(--accent); color:#fff; padding:8px 16px; border-radius:9px;
    font-weight:700; font-size:.9rem; }

/* ---- hero ---- */
#hero { text-align:center; padding:54px 16px 30px; }
#hero .pill { display:inline-block; background:#ecfdf5; color:#047857; font-weight:700;
    font-size:.78rem; letter-spacing:.04em; text-transform:uppercase;
    padding:6px 13px; border-radius:999px; border:1px solid #a7f3d0; }
#hero h1 { font-size:3rem; line-height:1.08; font-weight:800; letter-spacing:-1.5px;
    margin:18px 0 0; color:var(--ink); }
#hero h1 .hl { color:var(--accent); }
#hero p { font-size:1.15rem; color:var(--muted); max-width:560px; margin:16px auto 0; line-height:1.6; }

/* ---- app card ---- */
#appcard {
    border:1px solid var(--line) !important; border-radius:18px !important;
    box-shadow:0 10px 30px rgba(2,6,23,.06) !important; padding:22px !important;
    background:#fff !important; margin-top:14px;
}
#go-btn { background:var(--accent) !important; color:#fff !important; font-weight:700 !important;
    border:none !important; border-radius:10px !important; font-size:1rem !important; }
#go-btn:hover { filter:brightness(1.05); }

/* ---- verdict panel ---- */
.verdict { border-radius:14px; padding:30px 20px; text-align:center; min-height:236px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px solid var(--line); }
.verdict .icon { width:64px; height:64px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:32px; margin-bottom:14px; }
.verdict .title { font-size:1.6rem; font-weight:800; margin:2px 0; letter-spacing:-.4px; }
.verdict .sub { font-size:.98rem; color:var(--muted); font-weight:500; }
.verdict.idle { background:#f8fafc; border-style:dashed; }
.verdict.idle .icon { background:#eef2f7; }
.verdict.fresh { background:#f0fdf4; border-color:#bbf7d0; }
.verdict.fresh .icon { background:#dcfce7; }
.verdict.fresh .title { color:#15803d; }
.verdict.rotten { background:#fef2f2; border-color:#fecaca; }
.verdict.rotten .icon { background:#fee2e2; }
.verdict.rotten .title { color:#b91c1c; }

/* ---- how it works ---- */
#how { padding:52px 0 8px; }
#how h2 { text-align:center; font-size:1.7rem; font-weight:800; letter-spacing:-.6px; margin:0 0 6px; }
#how .lead { text-align:center; color:var(--muted); margin:0 0 26px; }
.step { border:1px solid var(--line); border-radius:14px; padding:22px; height:100%; background:#fff; }
.step .num { width:34px; height:34px; border-radius:9px; background:#ecfdf5; color:#047857;
    font-weight:800; display:flex; align-items:center; justify-content:center; margin-bottom:12px; }
.step h3 { font-size:1.05rem; font-weight:700; margin:0 0 6px; }
.step p { color:var(--muted); font-size:.92rem; line-height:1.55; margin:0; }

/* ---- footer ---- */
#foot { border-top:1px solid var(--line); margin-top:46px; padding:24px 4px;
    display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:.85rem; }
#foot .brand { font-weight:700; color:var(--ink); }
@media (max-width:640px){ #hero h1{font-size:2.1rem} #nav .links{display:none} }
"""

IDLE_HTML = """
<div class="verdict idle">
  <div class="icon">🍎</div>
  <div class="title">Awaiting your photo</div>
  <div class="sub">Upload an image and run the analysis to see results here.</div>
</div>
"""


def _verdict_html(label, confidence):
    pct = f"{confidence * 100:.0f}%"
    if label.lower().startswith("rotten") or "bad" in label.lower():
        return f"""
        <div class="verdict rotten">
          <div class="icon">⚠️</div>
          <div class="title">Spoiled</div>
          <div class="sub">{pct} confidence &middot; we recommend discarding this fruit.</div>
        </div>"""
    return f"""
    <div class="verdict fresh">
      <div class="icon">✓</div>
      <div class="title">Fresh</div>
      <div class="sub">{pct} confidence &middot; this fruit looks good to eat.</div>
    </div>"""


def classify(image):
    if image is None:
        return IDLE_HTML, {}
    if clf is None:
        return ("""<div class="verdict rotten"><div class="icon">⚙️</div>
                <div class="title">Model not found</div>
                <div class="sub">Run <code>python train.py --data-dir data</code> first.</div></div>""", {})
    _verdict, scores = clf.predict(Image.fromarray(image))
    top = max(scores, key=scores.get)
    return _verdict_html(top, scores[top]), scores


NAV = """
<div id="nav">
  <div class="brand"><span class="dot">🍃</span> FreshCheck</div>
  <div class="links"><span>How it works</span><span>Technology</span><span>About</span></div>
  <div class="cta">Get started</div>
</div>
"""

HERO = """
<div id="hero">
  <span class="pill">AI-powered freshness analysis</span>
  <h1>Is your fruit still <span class="hl">fresh</span>?</h1>
  <p>Upload a photo and our on-device model tells you in seconds whether it's
     good to eat or past its prime, no account, no cloud upload.</p>
</div>
"""

HOW = """
<div id="how">
  <h2>How it works</h2>
  <p class="lead">Three steps, instant results.</p>
</div>
"""

STEPS = [
    ("1", "Upload a photo", "Drag in or select a clear photo of a single piece of fruit."),
    ("2", "AI analysis", "A compact convolutional network inspects color and texture cues on your device."),
    ("3", "Get your verdict", "See a fresh / spoiled result with a confidence score in real time."),
]

# ------------------------------------------------------------------ layout ----
with gr.Blocks(title="FreshCheck: Fruit Freshness Tester") as demo:
    gr.HTML(NAV)
    gr.HTML(HERO)

    with gr.Row(equal_height=True, elem_id="appcard"):
        with gr.Column(scale=1):
            image_in = gr.Image(type="numpy", label="Fruit photo", height=300)
            go = gr.Button("Analyze freshness", elem_id="go-btn", size="lg")
        with gr.Column(scale=1):
            verdict_out = gr.HTML(IDLE_HTML)
            scores_out = gr.Label(label="Confidence breakdown", num_top_classes=2)

    sample_dirs = ["data/val/fresh", "data/val/rotten"]
    examples = []
    for d in sample_dirs:
        if os.path.isdir(d):
            examples += [[os.path.join(d, p)] for p in sorted(os.listdir(d))[:2]]
    if examples:
        gr.Examples(examples=examples, inputs=image_in, label="Sample images")

    gr.HTML(HOW)
    with gr.Row():
        for num, title, body in STEPS:
            gr.HTML(f'<div class="step"><div class="num">{num}</div>'
                    f'<h3>{title}</h3><p>{body}</p></div>')

    gr.HTML('<div id="foot"><span class="brand">🍃 FreshCheck</span>'
            '<span>TinyFreshNet · ~24k parameters · trained on real fruit photos</span></div>')

    go.click(classify, inputs=image_in, outputs=[verdict_out, scores_out])
    image_in.change(classify, inputs=image_in, outputs=[verdict_out, scores_out])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)

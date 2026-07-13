# 🍎 Fruit Freshness Tester

Upload a photo of a fruit and a small fine-tuned AI model tells you whether it's
**fresh** or has **gone bad**.

The model (`TinyFreshNet`) is a deliberately tiny convolutional neural network
(around **24,000 trainable parameters**) so it trains in minutes on a laptop
CPU and runs instantly at inference time.

## Live demo (in-browser, no backend)

The production app is a **static React SPA** that runs the model **entirely in
your browser** via `onnxruntime-web` (WebAssembly). The model is trained in
PyTorch, exported to ONNX, and shipped as a static asset, so there's no server,
no cold starts, and images never leave your device.

```bash
python export_onnx.py    # writes web/model.onnx + web/meta.json from the checkpoint
python -m http.server -d web 8000     # preview at http://localhost:8000
```

### Deploy to Vercel

The `web/` directory is a zero-build static site. To publish:

1. Push this repo to GitHub (already done).
2. In Vercel: **Add New → Project → Import** this repo.
3. Set **Root Directory = `web`**, Framework Preset = **Other**, leave the build
   command empty.
4. **Deploy.** Every push to `main` auto-redeploys.

(The `server.py` FastAPI backend is still included for local/server use, but the
deployed demo needs none of it.)

## Project layout

| File | What it does |
|------|--------------|
| `model.py` | The small CNN (`TinyFreshNet`) |
| `data_utils.py` | Image transforms + dataset loaders |
| `train.py` | Train the model and save a checkpoint |
| `inference.py` | Load the checkpoint, classify one image |
| `predict.py` | CLI: `python predict.py fruit.jpg` |
| `app.py` | Gradio web app (drag-and-drop upload) |
| `make_sample_data.py` | Generate synthetic data to test the pipeline |
| `prepare_real_data.py` | Convert the Kaggle dataset into the expected layout |

## The product: FreshCheck (vanilla JS + FastAPI)

The web app is a single static page built with **hand-written HTML, CSS, and
vanilla JavaScript**, no framework and no build step. It runs the model in the
browser with **onnxruntime-web**, so it needs no backend at all. A **FastAPI**
server is also included if you would rather run inference on a machine.

```bash
pip install -r requirements.txt
python train.py --data-dir data --epochs 40   # if checkpoints/freshnet.pt is missing
python evaluate.py                             # writes metrics.json (honest held-out test accuracy)
python server.py                               # serves http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000**. Endpoints:
- `GET /api/meta`: architecture, params, dataset sizes, measured train/val/test accuracy, device
- `POST /api/predict`: image → predicted class, confidence, full probabilities, inference latency, image metadata
- `GET /api/examples`: sample test images

The frontend (`web/index.html`) pulls in only **onnxruntime-web** from a CDN plus
two web fonts. Everything else is inline, so there's no `npm install` and the
page stays light. Design is a neo-brutalist produce-market look (thick ink
borders, hard offset shadows, acid green on cream, oversized Archivo type, a
price-burst accuracy sticker and a paper-receipt stats panel). Features:
drag-and-drop and camera capture, example images, an animated confidence gauge
and probability bars, a model-insights dashboard, batch analysis, copy JSON /
download report / shareable link, dark mode, and empty/loading/error states.

### Simpler Gradio version

A minimal Gradio UI also exists:

```bash
python make_sample_data.py        # synthetic data (only if you have no real data)
python train.py --data-dir data --epochs 12
python app.py                     # opens the Gradio web app
```

`make_sample_data.py` generates **synthetic** fruit images (bright = fresh,
brown blotches = rotten). This proves the whole pipeline runs and lets you play
with the app, but it will not judge real photos well. For that, use real data.

## Real data: banana set (already wired in, no auth)

The model currently shipped here is trained on **real banana photos** from the
Hugging Face dataset `nikibout/fresh-and-rotten-fruit` (~120 MB, no login). To
re-fetch and retrain from scratch:

```bash
python prepare_hf_banana.py          # downloads + lays out data/{train,val,test}/{fresh,rotten}
python train.py --data-dir data --epochs 40
python evaluate.py                   # reports held-out TEST accuracy
python app.py
```

`prepare_hf_banana.py` maps `freshbanana`->`fresh` and `rottenbanana`->`rotten`,
then builds a leak-free split. The source's Train and Test folders overlap
completely (every Train image is a byte-identical duplicate of a Test image), so
the raw ~510 file entries collapse to **300 unique images** (150 fresh, 150
rotten). The script de-duplicates by content hash, groups near-duplicate frames
of the same banana with a perceptual hash, and does a seeded, stratified,
group-aware **70/15/15 train/val/test** split so no near-duplicate straddles a
split boundary. `train.py` auto-applies inverse-frequency class weights and
selects the checkpoint on validation accuracy; the test split is never touched
during training.

Honest result on the held-out test set (44 images the model never saw):
**97.7% accuracy (43/44), 95% Wilson CI 88.2% to 99.6%**. The test set is small
(n=44), so that interval is wide. The earlier "100% validation accuracy" came
from exact-duplicate leakage across the old train/val split and has been removed.
The number is real for bananas on a small set, not a claim about every fruit; the
value of the project is the leak-free pipeline and the in-browser deployment,
which carry over to bigger datasets.

> Banana-only because local disk was too full (~1.3 GB free) for the full ~9 GB
> multi-fruit set. To go multi-fruit (apples/oranges/etc), free up disk and use
> `Densu341/Fresh-rotten-fruit` on Hugging Face, or the Kaggle route below.

## Use other real data (Kaggle multi-fruit)

1. Download the Kaggle dataset
   [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
   and unzip it.
2. Convert it to the binary fresh/rotten layout:
   ```bash
   python prepare_real_data.py --src /path/to/unzipped/dataset --dst data
   ```
3. Train and launch:
   ```bash
   python train.py --data-dir data --epochs 20
   python app.py
   ```

You can also just drop your own photos into:

```
data/train/fresh/*.jpg
data/train/rotten/*.jpg
data/val/fresh/*.jpg
data/val/rotten/*.jpg
data/test/fresh/*.jpg
data/test/rotten/*.jpg
```

## How it works

- Photos are resized to 64×64 and normalized.
- Three small conv blocks extract features; a global-average-pooled linear head
  outputs `fresh` vs `rotten` logits.
- Training uses augmentation (flips, rotation, color jitter) so the tiny net
  generalizes. The best validation checkpoint is saved to
  `checkpoints/freshnet.pt`.
- `inference.py` softmaxes the logits into probabilities and turns the top class
  into a human verdict.

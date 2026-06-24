# 🍎 Fruit Freshness Tester

Upload a photo of a fruit and a small fine-tuned AI model tells you whether it's
**fresh** or has **gone bad**.

The model (`TinyFreshNet`) is a deliberately tiny convolutional neural network
— around **24,000 trainable parameters** — so it trains in minutes on a laptop
CPU and runs instantly at inference time.

## Live demo (in-browser, no backend)

The production app is a **static React SPA** that runs the model **entirely in
your browser** via `onnxruntime-web` (WebAssembly). The model is trained in
PyTorch, exported to ONNX, and shipped as a static asset — so there's no server,
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

## The product: FreshCheck (React + FastAPI)

The polished web app is a single-page **React 18 + Tailwind + Framer Motion**
frontend served by a **FastAPI** backend that runs the PyTorch model.

```bash
pip install -r requirements.txt
python train.py --data-dir data --epochs 25   # if checkpoints/freshnet.pt is missing
python evaluate.py                             # writes metrics.json (honest train/val accuracy)
python server.py                               # serves http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000**. Endpoints:
- `GET /api/meta` — architecture, params, dataset sizes, measured train/val accuracy, device
- `POST /api/predict` — image → predicted class, confidence, full probabilities, inference latency, image metadata
- `GET /api/examples` — sample test images

The frontend (`web/index.html`) loads React/Framer Motion via import-map CDN and
Tailwind via CDN, so it runs with **no `npm install`** — handy when disk is tight.
Features: drag-and-drop + camera capture + example images, animated confidence
gauge and probability bars, an ML "Model Insights" dashboard, batch analysis,
prediction history, export JSON / download report / shareable link, dark mode,
and skeleton/empty/error states.

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

## Real data — banana set (already wired in, no auth)

The model currently shipped here is trained on **real banana photos** from the
Hugging Face dataset `nikibout/fresh-and-rotten-fruit` (~120 MB, no login). To
re-fetch and retrain from scratch:

```bash
python prepare_hf_banana.py          # downloads + lays out data/{train,val}/{fresh,rotten}
python train.py --data-dir data --epochs 25
python app.py
```

`prepare_hf_banana.py` pools the source's Train/Test images, maps
`freshbanana`->`fresh` and `rottenbanana`->`rotten`, and makes a seeded,
class-balanced 85/15 split. `train.py` auto-applies inverse-frequency class
weights to handle the fresh/rotten imbalance. Verified: 100% validation accuracy,
~99% confidence on held-out fresh/rotten bananas.

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

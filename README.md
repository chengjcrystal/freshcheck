# FreshCheck

Is that banana still good? Take a photo and a small neural network calls it
fresh or rotten, right in your browser. Nothing gets uploaded. The picture never
leaves your device.

The model (`TinyFreshNet`) is a deliberately tiny convolutional net, about
**24,000 parameters**, so it trains in a few minutes on a laptop CPU and answers
the moment you drop a photo in.

## Live demo (in-browser, no backend)

The deployed app is a static page that runs the model **entirely in your
browser** with `onnxruntime-web` (WebAssembly). It's trained in PyTorch,
exported to ONNX, and shipped as a plain asset, so there's no server, no cold
starts, and no upload.

```bash
python export_onnx.py                 # writes web/model.onnx + web/meta.json from the checkpoint
python -m http.server -d web 8000     # preview at http://localhost:8000
```

### Deploy to Vercel

The `web/` directory is a zero-build static site.

1. Push this repo to GitHub.
2. In Vercel: **Add New, Project, Import** this repo.
3. Set **Root Directory = `web`**, Framework Preset = **Other**, leave the build
   command empty.
4. **Deploy.** Every push to `main` redeploys on its own.

(The `server.py` FastAPI backend is still here for local or server use, but the
deployed demo needs none of it.)

## What's in here

| File | What it does |
|------|--------------|
| `model.py` | The small CNN (`TinyFreshNet`) |
| `data_utils.py` | Image transforms and dataset loaders |
| `train.py` | Train the model, save a checkpoint |
| `evaluate.py` | Measure honest train/val/test accuracy, write `metrics.json` |
| `inference.py` | Load the checkpoint, classify one image |
| `predict.py` | CLI: `python predict.py fruit.jpg` |
| `server.py` | FastAPI server (optional, for running inference on a machine) |
| `app.py` | Gradio web app (drag-and-drop upload) |
| `prepare_hf_banana.py` | Fetch and lay out the Hugging Face banana set |
| `prepare_real_data.py` | Convert the Kaggle multi-fruit dataset into this layout |
| `make_sample_data.py` | Synthetic data to test the pipeline end to end |

## The web app

The page is hand-written HTML, CSS, and vanilla JavaScript. No framework, no
build step. It runs the model in the browser with **onnxruntime-web**, so it
needs no backend. A **FastAPI** server is included if you'd rather run inference
on a machine.

```bash
pip install -r requirements.txt
python train.py --data-dir data --epochs 40   # only if checkpoints/freshnet.pt is missing
python evaluate.py                             # writes metrics.json (honest held-out test accuracy)
python server.py                               # serves http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000**. Endpoints:
- `GET /api/meta`: architecture, params, dataset sizes, measured train/val/test accuracy, device
- `POST /api/predict`: image to predicted class, confidence, full probabilities, latency, image metadata
- `GET /api/examples`: sample test images

The frontend (`web/index.html`) pulls in only **onnxruntime-web** from a CDN plus
two web fonts. Everything else is inline, so there's no `npm install` and the
page stays light. The look is a neo-brutalist produce-market style: thick ink
borders, hard offset shadows, acid green on cream, oversized Archivo type, a
price-burst accuracy sticker, and a paper-receipt stats panel. It does
drag-and-drop and camera capture, example photos, a confidence gauge and
probability bars, a four-level verdict (fresh, still ok, going off, rotten) that
tracks how sure the model is, batch analysis, copy JSON, download report, a
shareable link, dark mode, and the usual empty, loading, and error states.

### Simpler Gradio version

There's also a minimal Gradio UI:

```bash
python make_sample_data.py        # synthetic data, only if you have no real data
python train.py --data-dir data --epochs 12
python app.py                     # opens the Gradio web app
```

`make_sample_data.py` draws **synthetic** fruit (bright = fresh, brown blotches =
rotten). It proves the pipeline runs and lets you play with the app, but it won't
judge real photos well. For that, use real data.

## Real data: the banana set (already wired in, no auth)

The shipped model is trained on **real banana photos** from the Hugging Face
dataset `nikibout/fresh-and-rotten-fruit` (~120 MB, no login). To refetch and
retrain:

```bash
python prepare_hf_banana.py          # downloads + lays out data/{train,val,test}/{fresh,rotten}
python train.py --data-dir data --epochs 40
python evaluate.py                   # reports held-out TEST accuracy
python app.py
```

`prepare_hf_banana.py` maps `freshbanana` to `fresh` and `rottenbanana` to
`rotten`, then builds a leak-free split. Here's the catch worth knowing about:
the source's Train and Test folders overlap completely. Every Train image is a
byte-for-byte copy of a Test image, so the raw ~510 files collapse to **300
unique images** (150 fresh, 150 rotten). The script de-duplicates by content
hash, groups near-identical frames of the same banana with a perceptual hash,
and does a seeded, stratified, group-aware **70/15/15** split so no look-alike
frame straddles the boundary. `train.py` applies inverse-frequency class weights
and picks the checkpoint on validation accuracy. The test split is never touched
during training.

Honest result on the held-out test set (44 images the model never saw):
**97.7% accuracy (43/44), 95% Wilson CI 88.2% to 99.6%**. That set is small
(n=44), so the interval is wide. The old "100% validation accuracy" came from
exact-duplicate leakage across the previous train/val split and is gone. The
number is real for bananas on a small set, not a promise about every fruit. The
point of the project is the leak-free pipeline and the in-browser deployment,
which carry over to bigger datasets.

> Banana-only because local disk was too tight (~1.3 GB free) for the full ~9 GB
> multi-fruit set. To go multi-fruit (apples, oranges, and so on), free up disk
> and use `Densu341/Fresh-rotten-fruit` on Hugging Face, or the Kaggle route
> below.

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

- Photos get resized to 64x64 and normalized.
- Three small conv blocks pull out features; a global-average-pooled linear head
  outputs `fresh` vs `rotten` logits.
- Training uses augmentation (flips, rotation, color jitter) so the tiny net
  generalizes. The best validation checkpoint is saved to
  `checkpoints/freshnet.pt`.
- At inference the logits go through softmax, and the top class plus its
  confidence becomes one of four plain verdicts.

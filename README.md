# Histopathology Binary Classification under Distribution Shift

Binary classification (tumor vs. non-tumor) of histopathology image patches, with a focus on robustness to **staining variations across hospital centers**. Train data comes from 3 centers, validation from a 4th, and test from a 5th — meaning the model must generalize across staining protocols it has never seen.

**Best validation accuracy: 0.976**

---

## Approach

The key insight of this version is that the **backbone choice matters far more than classifier complexity**. Earlier versions stacked sophisticated tricks (multi-head ensembles, attention fusion, focal loss, CutMix) on top of a DINOv2 backbone pre-trained on natural images, with limited gains. Switching to a backbone pre-trained on histopathology data was the single biggest improvement.

### Backbone: Phikon-v2 + DINOv2 ensemble

- **Phikon-v2** (Owkin, *Nature Medicine* 2024): a ViT-L pre-trained with the DINOv2 self-supervised method on **450 million histopathology patches** sampled from 60,000 whole-slide images. Its features are natively calibrated for tissue analysis and intrinsically robust to staining shift, since it has seen hundreds of staining protocols during pre-training.
- **DINOv2 ViT-L/14** (Meta): a generalist self-supervised vision transformer pre-trained on 142 million natural images. Captures broad visual structure that complements Phikon's domain-specific features.

The features from both backbones (1024 + 1024) are concatenated into a 2048-dimensional vector per image. Both backbones are kept **frozen** — only a small MLP classifier on top is trained.

### Classifier

A deliberately simple MLP: `Linear(2048 → 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256 → 1)`. With features this strong, complex classifiers tend to overfit rather than help.

### Training tricks

- **Multi-pass augmented feature extraction**: each training image is passed through the backbone 3 times with different stain augmentations (color jitter, flips, grayscale, blur). The resulting feature dataset is 3× larger and exposes the classifier to varied stain appearances of every patch.
- **BCEWithLogitsLoss + label smoothing** (0.05) to reduce overconfidence on training centers.
- **AdamW + cosine annealing** for stable optimization.
- **Mixup** (α = 0.2) on features to smooth decision boundaries.
- **Stochastic Weight Averaging** (SWA): after epoch 20, model weights are averaged over the remaining epochs. The averaged weights sit in flat loss minima, which generalize better across distributions.
- **Optimal threshold search**: instead of the fixed 0.5 cutoff, the threshold is tuned on the validation set to maximize accuracy.

### Test-Time Augmentation (TTA)

At inference, each test image is passed through the full pipeline 5 times: 1 clean pass + 4 augmented passes with moderate color jitter. The resulting probabilities are averaged before thresholding. This smooths out staining-related noise on the unseen test center.

---

## Project structure

```
.
├── config.py              # All hyperparameters and paths
├── main.py                # Pipeline orchestration + CLI
├── model.py               # Phikon-v2 + DINOv2 loading, classifier head
├── feature_extraction.py  # Multi-pass feature extraction with disk cache
├── train.py               # Training loop with SWA, mixup, label smoothing
├── predict.py             # TTA prediction on test set
├── transforms.py          # Stain augmentation pipelines
├── dataset.py             # H5 dataset loaders
├── utils.py               # Seeding, device, checkpointing, threshold search
├── requirements.txt
├── data/                  # train.h5, val.h5, test.h5 (not included)
├── checkpoints/           # Saved model weights
├── cache/                 # Pre-extracted features (~GB-scale)
└── submission.csv         # Final predictions
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `torchvision>=0.15`, `transformers>=4.30`, `h5py`, `numpy`, `pandas`, `scikit-learn`, `tqdm`.

The first run downloads Phikon-v2 weights from HuggingFace (~1.2 GB) and DINOv2 ViT-L weights from torch.hub (~1.1 GB). These are cached locally and only downloaded once.

---

## Usage

Place the H5 data files in the `data/` directory, then:

```bash
# Full pipeline: extract features → train → predict
python main.py

# Resume training from the last checkpoint
python main.py --resume

# Predict only (uses checkpoints/best_model.pth)
python main.py --predict-only

# Predict without TTA (faster, slightly less accurate)
python main.py --predict-only --no-tta

# Predict with a custom threshold
python main.py --predict-only --threshold 0.48

# Use the SWA model instead of best checkpoint
python main.py --predict-only --model swa

# Force re-extraction of features (clears the cache)
python main.py --clear-cache
```

The output `submission.csv` contains binary predictions (`ID,Pred`). A companion file `submission_probs.csv` contains the raw averaged probabilities, useful for post-hoc threshold tuning without re-running prediction.

---

## Key configuration

Edit `config.py` to adjust:

```python
BACKBONE = "owkin/phikon-v2"        # Histopathology foundation model
USE_DINOV2_ENSEMBLE = True          # Concatenate DINOv2 features
N_AUG_PASSES = 3                    # Augmented feature extraction passes
HIDDEN_DIMS = [256]                 # Classifier MLP architecture
LR = 1e-3
LABEL_SMOOTHING = 0.05
USE_MIXUP = True
USE_SWA = True
TTA_RUNS = 5
```

---

## Caching behavior

Feature extraction is the expensive part of the pipeline (multiple forward passes through two ViT-L backbones on 100k+ images). Extracted features are cached to `cache/` and reused across runs. Cache files are keyed by backbone names and number of augmentation passes, so changing those settings invalidates the cache automatically.

If you change augmentation strength or other extraction-time settings, run `python main.py --clear-cache` to force re-extraction.

---

## Approaches that didn't work

For context, several promising-looking ideas were tested before settling on this version:

- **Multi-backbone with attention fusion** (DINOv2 ViT-S + ViT-B + ViT-L with learned per-image weights) — added complexity without measurable gain. The bottleneck was the backbone domain, not how features were combined.
- **Multi-head ensemble + Focal Loss + CutMix on features** — overengineered the classifier when the issue was upstream.
- **K-Fold ensemble** of 5 classifiers on stratified splits — the MLP converges so consistently that the 5 folds become highly correlated and averaging them brings nothing.
- **End-to-end LoRA fine-tuning of Phikon-v2** — the most theoretically promising direction (adapting the backbone to the specific task with ~0.5M trainable parameters), but infeasible on consumer GPUs: requires more than 8 GB of VRAM to train at a reasonable batch size.

---

## References

- Filiot, A. et al. *Phikon-v2, A large and public feature extractor for biomarker prediction*. arXiv:2409.09173 (2024).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193 (2024).
- Izmailov, P. et al. *Averaging Weights Leads to Wider Optima and Better Generalization* (SWA). UAI 2018.
- Zhang, H. et al. *mixup: Beyond Empirical Risk Minimization*. ICLR 2018.

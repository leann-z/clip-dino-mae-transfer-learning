# Visual Representation Learning: CLIP, DINO & MAE

Comparison of three self-supervised ViT backbones, CLIP, DINO, and MAE, on a 10-class fruit classification task. Features are extracted and evaluated via t-SNE visualisation, linear probing, and end-to-end fine-tuning to understand what each pre-training objective actually learns.

---

##  Results

| Model | Linear Probe Acc | Fine-tune Acc |
|---|---|---|
| CLIP | **92.4%** | 90.0% |
| DINO | 87.8% | 89.6% |
| MAE | 68.6% | **86.4%** |

> CLIP has the strongest frozen representations (best linear probe). DINO produces the most visually separated t-SNE clusters. MAE benefits most from fine-tuning (+17.8%) as its reconstruction-based features are poorly suited for classification until adapted.

---

## ️ Repository Structure

```
.
├── README.md
├── evaluate.py              # Feature extraction, linear probe, fine-tuning pipeline
├── representations.py       # Feature extractor, FeaturesDataset, t-SNE, training utilities
│
└── figures/
    ├── part1_features_t-sne_(clip).png                        # t-SNE plots for each backbone
    ├── part1_features_t-sne_(dino).png
    ├── part1_features_t-sne_(mae).png
    ├── part1_mae_banana_lemon_overlap.png               # Class overlap density plots (banana vs lemon)
    ├── clip_linearprobe.png           # Training loss curves
    ├── dino_linearprobe.png
    ├── mae_linerprobe.png
    ├── finetune_clip.png
    ├── finetune_dino.png
    └── finetune_mae.png
```

---

## Methodology

### Backbones
All three use `ViT-B/16` as the base architecture, loaded via `timm`, but differ in pre-training objective:

| Model | Pre-training Objective | Key Characteristic |
|---|---|---|
| CLIP | Contrastive image-text alignment | Learns semantic + visual similarity |
| DINO | Self-distillation (teacher-student, no labels) | Learns pure visual structure |
| MAE | Masked patch reconstruction | Learns low-level texture and pixel features |

### Evaluation Protocol

**t-SNE** — Features from the full training set (13,000 images) are projected to 2D to visualise cluster structure and separability.

**Linear probing** — Backbone frozen, a single linear layer trained on top for 32 epochs with AdamW. Tests the linear separability of the frozen feature space.

**Fine-tuning** — Backbone unfrozen with a low learning rate (1e-5), head initialised from the linear probe weights. Tests how well representations adapt to the downstream task.

### Key Findings

- **CLIP** leads on linear probing despite having more t-SNE cluster overlap than DINO — t-SNE captures local neighbourhood structure, not global linear separability.
- **DINO** produces the cleanest cluster separation in t-SNE, particularly for visually distinct classes (jackfruit, pineapple). Its teacher-student objective learns structural features without semantic priors.
- **MAE** confuses visually similar classes at the feature level — banana and lemon cluster together due to shared yellow pixel statistics. Fine-tuning corrects this by redirecting the objective from reconstruction to classification.
- **Catastrophic forgetting**: CLIP's fine-tune accuracy drops slightly from its linear probe (90% vs 92.4%), consistent with the well-documented degradation of large pre-trained models on small downstream datasets.

---

## ️ Setup & Usage

```bash
pip install torch timm scikit-learn matplotlib gdown tqdm
```

```bash
python evaluate.py
```

This will automatically download the dataset, extract features for all three backbones, produce t-SNE plots and density visualisations, train and evaluate linear probes, and run fine-tuning, saving all figures to `figures/`.

---

## Report

Full analysis with figures is in [`report.pdf`](./report.pdf).

## Acknowledgements

The feature extraction backbone and dataset pipeline were provided by the University of Cambridge Department of Engineering as part of a graduate advanced computer vision course. Original framework developed by Ayush Tewari and Elliot Wu. All experimental analysis, visualisations, and findings are my own work.

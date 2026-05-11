# Faithfulness Estimation of Conditional Generative Models

Companion code for experiments on per-sample trust scoring and conditional generation under compositional shift.

The trust score decomposes into a **global realism** term (Mahalanobis distance to real features) and an **attribute-wise faithfulness** term (per-attribute margin against competing values). It is post-hoc, requires no retraining, and works with any pretrained conditional diffusion model plus an external feature encoder. A during-generation variant maps internal denoising states into the same scoring space via a shallow translator, enabling early abstention.

## Installation

```bash
pip install uv
uv sync          # installs from pyproject.toml + uv.lock
```

All commands assume `PYTHONPATH=src uv run python ...`.

## Reproduction

### 1. Train a model

```bash
PYTHONPATH=src uv run python scripts/run_experiment.py \
    --config-name=config_rxrx1_marginal           # or config_celeba_marginal, config_rxrx1, ...
```

Variant configs (`configs/variant/`): `vanilla`, `repa`, `repa_siglip`, `repa_openphenom`.

### 2. Generate samples

```bash
PYTHONPATH=src uv run python scripts/generate_samples.py \
    --config-name=generate_samples_rxrx1 \
    checkpoint_key=rxrx1_repa_siglip_marginal_v1
```

Use `scripts/generate_samples_repa.py` for REPA-aligned models.

### 3. Cache features

```bash
PYTHONPATH=src uv run python scripts/cache_features.py --config-name=cache_rxrx1
```

### 4. Trust evaluation (post-generation)

```bash
PYTHONPATH=src uv run python scripts/run_trust_evaluation.py --help
```

### 5. During-generation translator (post-hoc alignment)

```bash
# Extract raw hidden states across timesteps
PYTHONPATH=src uv run python scripts/posthoc_alignment/extract_raw_hidden_rxrx1.py ...

# Train mapper (whitened-geometry loss)
PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
    --config-path=../../configs/posthoc_alignment \
    --config-name=train_mapper_rxrx1_siglip_whitened

# Evaluate
PYTHONPATH=src uv run python scripts/posthoc_alignment/eval_posthoc_alignment_rxrx1.py ...
```

### 6. CellProfiler validation (RxRx1, Table 2 / Figure 2)

```bash
PYTHONPATH=src uv run python scripts/cp_morphology_validation.py
PYTHONPATH=src uv run python scripts/cp_decile_binning.py
PYTHONPATH=src uv run python scripts/cp_decile_selection_plots.py
```

### 7. Aggregate paper tables / figures

```bash
PYTHONPATH=src uv run python scripts/aggregate_fpr95_delta_kid.py
PYTHONPATH=src uv run python scripts/aggregate_scorer_ablations.py
PYTHONPATH=src uv run python scripts/plot_posthoc_timestep_story.py
PYTHONPATH=src uv run python scripts/plot_trust_vs_kid_scatter.py
```

## Checkpoints

`configs/checkpoints.yaml` is the canonical registry. The 14 paper-ready checkpoints (vanilla + REPA-DINOv3 + REPA-SigLIP for both CelebA and RxRx1, both full-support and support-shift, plus REPA-OpenPhenom for RxRx1) are all flagged `paper_ready: true`. Resolve a path with:

```python
from faithful_cond_gen.utils.checkpoints import get_checkpoint_path
ckpt = get_checkpoint_path("rxrx1_repa_siglip_marginal_v1")
```

## Layout

```
src/faithful_cond_gen/
├── data/                    # CelebA + RxRx1 dataset loaders
├── model/                   # SiT backbone, REPA encoder, EMA, generator
├── pl_modules/              # PyTorch Lightning wrapper
├── posthoc_alignment/       # during-gen translator (mapper, dataset, losses)
├── eval/
│   ├── scoring/             # 9 scorers (cosine, knn, mahalanobis, ...)
│   └── trust_eval/          # main evaluation pipeline + ablation
│       └── eval_layers/     # ranking, binning, fpr95, failure detection, downstream, OOD
└── utils/                   # config, checkpoints, metrics, flops

scripts/
├── run_experiment.py            # train
├── generate_samples{,_repa}.py  # sample
├── cache_features.py            # feature extraction
├── run_trust_evaluation.py      # post-gen eval
├── posthoc_alignment/           # during-gen translator pipeline
├── cp_*.py                      # CellProfiler validation (RxRx1)
├── aggregate_*.py               # table aggregation
├── plot_*.py                    # figure generation
├── analyze_*.py                 # analysis utilities
├── diag_openphenom_*.py         # OpenPhenom failure diagnostic (appendix)
└── diag_posthoc_*.py            # during-gen story diagnostics

configs/
├── config_{celeba,rxrx1}{,_marginal}.yaml
├── generate_samples_{celeba,rxrx1}.yaml
├── cache_{celeba,rxrx1}.yaml
├── extract_raw_hidden{,_rxrx1}.yaml
├── checkpoints.yaml
├── variant/                 # vanilla, repa, repa_siglip, repa_openphenom, ...
├── dataset/                 # celeba, rxrx1, marginal variants
├── model/                   # generator + encoder configs
├── train/                   # training hyperparameters
└── posthoc_alignment/       # 12 mapper training configs
```

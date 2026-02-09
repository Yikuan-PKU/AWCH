# AW-CH: On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature

Code for the ICML 2026 paper: *"On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature"*.

## Overview

This repository provides the implementation for studying the relationship between the SGD noise covariance $\mathbf{C}$ and the loss Hessian $\mathbf{H}$ in neural networks, based on the **Activity–Weight Duality (AWD)** framework ([Feng et al., 2023](https://arxiv.org/abs/2308.05286)).

**Key findings:**
- The noise covariance is governed by the **second moment** of per-sample Hessians: $\mathbf{C} \propto \mathbb{E}_p[\mathbf{h}_p^2]$, where $\mathbf{H} = \mathbb{E}_p[\mathbf{h}_p]$.
- $\mathbf{C}$ and $\mathbf{H}$ approximately commute ($[\mathbf{C}, \mathbf{H}] \approx 0$) rather than coincide.
- Their diagonal elements follow a **power-law** relation $C_{ii} \propto H_{ii}^{\gamma}$ with a theoretically bounded exponent $1 \leq \gamma \leq 2$.
- Cross-entropy (CE) loss exhibits superlinear scaling ($\gamma > 1$, up to ~1.4), while MSE loss yields approximately linear scaling ($\gamma \approx 1$).

## Method

### Activity–Weight Duality (AWD)

For a fully connected layer with weights $\mathbf{W}$, given an input activity perturbation $\Delta \bm{a}$ from a matched sample pair, the **Minimal AWD** finds the weight perturbation $\Delta \mathbf{W}^*$ that preserves pre-activations with minimal Frobenius norm:

$$\Delta \mathbf{W}^* = \frac{(\mathbf{W} \Delta \bm{a}) \bm{a}^\top}{\|\bm{a}\|^2}$$

### AWD-Based Noise Covariance Decomposition

Under the AWD gradient approximation, the gradient difference between two mini-batches is dominated by the Hessian-driven term near convergence:

$$\bm{g}_{\mu\nu} \approx \frac{1}{B} \sum_{p \in \mathcal{B}_\nu} \mathbf{h}_p(\bm{w}) \Delta \bm{w}_p^{\mu\nu}$$

This yields the noise covariance as a quadratic form of the per-sample Hessian (Theorem 1 in the paper):

$$C_{ij} \approx \frac{\sigma_w^2}{2B} \mathbb{E}_p \left[ \sum_m (\kappa_m^{(p)})^2 (\bm{u}_m^{(p)} \cdot \bm{v}_i)(\bm{u}_m^{(p)} \cdot \bm{v}_j) \right]$$

where $\kappa_m^{(p)}$ and $\bm{u}_m^{(p)}$ are the eigenvalues and eigenvectors of the per-sample Hessian $\mathbf{h}_p$, and $\{\bm{v}_i\}$ is the global Hessian eigenbasis.

### Code Decomposition

In the code, the AWD-based covariance is decomposed into three components corresponding to the full expansion (Eq. 10 in the paper):

| Code Variable | Paper Notation | Formula |
|---------------|----------------|---------|
| `C1` / `C1_dia` | $\mathbf{C}^{hh}$ | $\mathbb{E}[\mathbf{h}_p \Delta\bm{w}_p \Delta\bm{w}_p^\top \mathbf{h}_p^\top]$ — pure Hessian–weight contribution |
| `C2` / `C2_dia` | $\mathbf{C}^{hg}$ | Cross-interaction between Hessian-weight (Term I) and gradient-activity (Term II) |
| `C3` / `C3_dia` | $\mathbf{C}^{gg}$ | $\mathbb{E}[(\nabla\Delta\bm{w}_p)^\top \nabla\ell_p \cdot (\nabla\Delta\bm{w}_p)^\top \nabla\ell_p^\top]$ — pure gradient contribution |

- `*_dia` variants: diagonal terms (same-sample pairs, $p = q$)
- Without `_dia`: full terms including cross-sample contributions

Additional stored quantities:
| Code Variable | Description |
|---------------|-------------|
| `C1_dia_w_dia` | $C_1$ diagonal with only diagonal elements of the weight perturbation covariance $\mathcal{M}_p$ |
| `C1_h` | Hessian second moment $\mathbb{E}_p[\mathbf{h}_p^2]$ (with $\mathcal{M}_p$ replaced by identity) |
| `H_1_d` | Diagonal of first Hessian moment: $H_{ii} = \bm{v}_i^\top \mathbb{E}_p[\mathbf{h}_p] \bm{v}_i$ |
| `H_2_d` | Diagonal of second Hessian moment: $\bm{v}_i^\top \mathbb{E}_p[\mathbf{h}_p^2] \bm{v}_i$ |
| `Covar` | Empirical noise covariance via Eq. 2: $\mathbf{C} = \frac{1}{B}[\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^\top - \bm{g}\bm{g}^\top]$ |
| `Hessian` | Global Hessian $\mathbf{H} = \nabla^2 \mathcal{L}$ and its eigen-decomposition (`components`) |

All matrices are computed and stored in the **Hessian eigenbasis** $\{\bm{v}_i\}$ (referred to as `components` in the code).

## Project Structure

```
├── model_config.py          # Hyperparameter configuration and sweep ranges
├── data.py                  # Data loading (MNIST / CIFAR-10), subset sampling, label noise injection
├── models.py                # Model definitions (FC / MLP / CNN) with intermediate feature caching
├── train_model.py           # Training with SGD + CosineAnnealingLR, gradient clipping, early stopping
├── utils.py                 # Hessian eigen-decomposition, Fisher information, noise covariance (Eq. 2)
├── AWD_cuda.py              # Core AWD: per-sample Hessian h_p, gradient, C matrix computation (Theorem 1)
│
├── cal_C_cuda_multi.py      # Entry point: multi-run pipeline (train → Hessian → C matrices)
├── cal_h_g.py               # Entry point: per-sample Hessian h_p & gradient computation only
│
├── Figures.ipynb            # Visualization: C-H commutativity, log-log power-law plots (Fig. 1-4)
└── effective_power.ipynb    # Effective power / suppression experiment analysis (Fig. 5)
```

## Pipeline

```
python cal_C_cuda_multi.py --max_e 100 --net_size 50 --n_class 10 --runs 5
  │
  ├─ 1. train_model.train()            # Train model (SGD + CosineAnnealing)
  ├─ 2. utils.cal_hessian_cuda()       # Global Hessian H → eigen-decomposition → {v_i} (components)
  ├─ 3. utils.cal_noise_covar_*()      # Empirical noise covariance C (Eq. 2 in paper)
  ├─ 4. AWD_cuda.cal_C_cuda()          # AWD-based C1, C2, C3 decomposition (Theorem 1)
  └─ 5. torch.save()                   # Save to AWCH_data/
```

For per-sample Hessian $\mathbf{h}_p$ & gradient only:
```
python cal_h_g.py --max_e 100
```

## Reproducing Paper Results

The key experiments in the paper can be reproduced as follows:

| Paper Content | Code Entry |
|---------------|------------|
| Table 1 ($\gamma_\text{emp}$ vs $\gamma_\text{AWD}$) | `cal_C_cuda_multi.py` with different `--n_class`, loss functions, and model/dataset combinations |
| Fig. 1 (C-H commutativity) | `Figures.ipynb` — C-H commutativity section |
| Fig. 3 (Log-log power law) | `Figures.ipynb` — log-log plot section |
| Fig. 5 (Suppression experiment) | `effective_power.ipynb` |
| Per-sample Hessian $\mathbf{h}_p$ statistics | `cal_h_g.py` |

## Command-Line Arguments

### cal_C_cuda_multi.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_e` | int | 200 | Maximum number of training epochs |
| `--net_size` | int | 50 | Hidden layer width |
| `--n_class` | int | 10 | Number of classification classes |
| `--runs` | int | 5 | Number of repeated runs with different random seeds |

### cal_h_g.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_e` | int | 200 | Maximum number of training epochs |

## Configuration Parameters (model_config.py)

| Parameter | Type | Default | Description | Paper Reference |
|-----------|------|---------|-------------|-----------------|
| `alpha` | float | 0.1 | SGD learning rate $\eta$ | Eq. 1 |
| `lss_fn` | str | `'mse'` | Loss function: `'mse'` (Softmax+MSE) / `'cse'` (CrossEntropy) / `'lmse'` (Linear MSE) | Section 4, Table 1 |
| `B` | int | 50 | Mini-batch size $B$ | Eq. 1–2 |
| `train_size` | int | 2000 | Training samples per class | — |
| `test_size` | int | 1000 | Test set size | — |
| `rho` | float | 0 | Label noise probability | — |
| `net_size` | int | 50 | Hidden layer width | — |
| `s` | int | 1 | Weight initialization scaling factor | — |
| `d` | float | 0 | Dropout probability | — |
| `beta` | float | 0 | L2 regularization coefficient | — |
| `stop_loss` | float | 1e-5 | Early stopping loss threshold | — |
| `sample_holder` | list | [0..9] | Class IDs for matched sample pair construction | Section 3.1 |
| `class_number` | int | 10 | Total number of classes $\mathcal{C}$ | Table 1 |
| `layer_index` | list | [1] | Target layer index for AWD analysis (FC/MLP: [1]; CNN: [8]) | Section 3.2 |
| `dataset` | str | — | Dataset: `'mnist'` / `'cifar10'` / `'fdata'` | Table 1 |
| `model` | str | — | Architecture: `'FC'` / `'MLP'` / `'CNN'` | Table 1 |

### Hyperparameter Sweep Ranges

| Parameter | Values |
|-----------|--------|
| `alpha` | 0.005, 0.01, 0.02, 0.05, 0.1 |
| `train_size` | 400, 800, 1600, 2000, 3200, 5000 |
| `rho` | 0, 0.091, 0.13, 0.167, 0.2 |
| `s` | 1, 4, 5, 6, 7 |
| `d` | 0, 0.05, 0.1, 0.2, 0.3 |
| `beta` | 0, 5e-3, 1e-2, 2e-2 |

## Supported Models

| Model | Architecture | Target Layer |
|-------|-------------|--------------|
| FC | Flatten → Linear(784,H) → ReLU → Linear(H,H) → ReLU → Linear(H,10) | [1] |
| MLP | Linear → ReLU → Dropout → Linear → ReLU → Linear → ReLU → Linear(10) | [1] |
| CNN | VGG-style Conv [32,'M',64,'M',128,128,'M'] → AdaptiveAvgPool → FC(128,20) → ReLU → FC(20,C) | [8] |

All models use **no bias** and cache intermediate layer features in `self.feature` for Hessian computation.

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| MNIST | Handwritten digits, 28×28 grayscale |
| CIFAR-10 | 10-class natural images, 32×32 color (normalized) |
| fdata | Synthetic dataset |

## Getting Started

### Requirements

- Python 3.8+
- PyTorch (with CUDA support)
- torchvision
- numpy, matplotlib, scipy

### Installation

```bash
pip install torch torchvision numpy matplotlib scipy
```

### Run

```bash
# Compute C matrices with 5 random seeds
python cal_C_cuda_multi.py --max_e 100 --net_size 50 --n_class 10 --runs 5

# Compute per-sample Hessian and gradients
python cal_h_g.py --max_e 100
```

## Output Data Format

Results are saved under `AWCH_data/` with the naming convention:

```
AWCH_data/NS{net_size}_TrainSize{train_size}_SampleN{sample_number}_ClassN{n_class}_B{batch}lr{lr}_lossfn_{loss}_model_{model}_dataset_{dataset}/
  ├── C_epoch_{epoch}_run_{id}.pt          # Full AWD decomposition results
  └── h_g_tensors_holder_epoch_{epoch}.pt  # Per-sample Hessian & gradient tensors
```

**Contents of `C_epoch_*.pt`:**

| Key | Description | Paper Reference |
|-----|-------------|-----------------|
| `C1_dia`, `C1` | $\mathbf{C}^{hh}$: Hessian–weight contribution (diagonal / full) | Eq. 10, Term I × Term I |
| `C2_dia`, `C2` | $\mathbf{C}^{hg}$: Cross-interaction (diagonal / full) | Eq. 10, Term I × Term II |
| `C3_dia`, `C3` | $\mathbf{C}^{gg}$: Gradient–activity contribution (diagonal / full) | Eq. 10, Term II × Term II |
| `C1_dia_w_dia` | $\mathbf{C}^{hh}$ with diagonal-only $\mathcal{M}_p$ | Isotropy check (Theorem 1) |
| `C1_h` | $\mathbb{E}_p[\mathbf{h}_p^2]$ with $\mathcal{M}_p = \mathbf{I}$ | Eq. 8 (core result) |
| `H_1_d` | $H_{ii}$ — diagonal of first Hessian moment | $\mathbf{H} = \mathbb{E}_p[\mathbf{h}_p]$ |
| `H_2_d` | Diagonal of second Hessian moment $\mathbb{E}_p[\mathbf{h}_p^2]$ | Theorem 1 |
| `C` / `Covar` | Empirical noise covariance | Eq. 2 |
| `Hessian` | Global Hessian $\mathbf{H}$ and eigen-decomposition | — |

## Citation

```bibtex
@inproceedings{zhang2026superlinear,
  title={On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature},
  author={Zhang, Yikuan and Yang, Ning and Tu, Yuhai},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

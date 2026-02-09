# AW-CH: On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature

Code for the paper: *"On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature"*.

## Overview

This repository provides the implementation for studying the relationship between the SGD noise covariance $\mathbf{C}$ and the loss Hessian $\mathbf{H}$ in neural networks, based on the **Activity–Weight Duality (AWD)** framework.

**Key findings:**
- The noise covariance is governed by the **second moment** of per-sample Hessians: $\mathbf{C} \propto \mathbb{E}_p[\mathbf{h}_p^2]$, where $\mathbf{H} = \mathbb{E}_p[\mathbf{h}_p]$.
- $\mathbf{C}$ and $\mathbf{H}$ approximately commute ($[\mathbf{C}, \mathbf{H}] \approx 0$) rather than coincide.
- Their diagonal elements follow a **power-law** relation $C_{ii} \propto H_{ii}^{\gamma}$ with a theoretically bounded exponent $1 \leq \gamma \leq 2$.
- Cross-entropy (CE) loss exhibits superlinear scaling ($\gamma > 1$, up to ~1.4), while MSE loss yields approximately linear scaling ($\gamma \approx 1$).

## Method

### Activity–Weight Duality (AWD)

For a fully connected layer with weights $\mathbf{W}$, given an input activity perturbation $\Delta \mathbf{a}$ from a matched sample pair, the **Minimal AWD** finds the weight perturbation $\Delta \mathbf{W}^*$ that preserves pre-activations with minimal Frobenius norm:

$$\Delta \mathbf{W}^* = \frac{(\mathbf{W} \Delta \mathbf{a}) \mathbf{a}^\top}{\|\mathbf{a}\|^2}$$

### AWD-Based Noise Covariance Decomposition

Under the AWD gradient approximation, the gradient difference between two mini-batches is dominated by the Hessian-driven term near convergence:

$$\mathbf{g}_{\mu\nu} \approx \frac{1}{B} \sum_{p \in \mathcal{B}_\nu} \mathbf{h}_p(\mathbf{w}) \Delta \mathbf{w}_p^{\mu\nu}$$

This yields the noise covariance as a quadratic form of the per-sample Hessian (Theorem 1 in the paper):

$$C_{ij} \approx \frac{\sigma_w^2}{2B} \mathbb{E}_p \left[ \sum_m (\kappa_m^{(p)})^2 (\mathbf{u}_m^{(p)} \cdot \mathbf{v}_i)(\mathbf{u}_m^{(p)} \cdot \mathbf{v}_j) \right]$$

where $\kappa_m^{(p)}$ and $\mathbf{u}_m^{(p)}$ are the eigenvalues and eigenvectors of the per-sample Hessian $\mathbf{h}_p$, and $\{\mathbf{v}_i\}$ is the global Hessian eigenbasis.

### Code Decomposition

In the code, the AWD-based covariance is decomposed into three components corresponding to the full expansion (Eq. 10 in the paper):

| Code Variable | Paper Notation | Formula |
|---------------|----------------|---------|
| `C1` / `C1_dia` | $\mathbf{C}^{hh}$ | $\mathbb{E}[\mathbf{h}_p \Delta\mathbf{w}_p \Delta\mathbf{w}_p^\top \mathbf{h}_p^\top]$ — pure Hessian–weight contribution |
| `C2` / `C2_dia` | $\mathbf{C}^{hg}$ | Cross-interaction between Hessian-weight (Term I) and gradient-activity (Term II) |
| `C3` / `C3_dia` | $\mathbf{C}^{gg}$ | $\mathbb{E}[(\nabla\Delta\mathbf{w}_p)^\top \nabla\ell_p \cdot (\nabla\Delta\mathbf{w}_p)^\top \nabla\ell_p^\top]$ — pure gradient contribution |

- `*_dia` variants: diagonal terms (same-sample pairs, $p = q$)
- Without `_dia`: full terms including cross-sample contributions

Additional stored quantities:
| Code Variable | Description |
|---------------|-------------|
| `C1_dia_w_dia` | $C_1$ diagonal with only diagonal elements of the weight perturbation covariance $\mathcal{M}_p$ |
| `C1_h` | Hessian second moment $\mathbb{E}_p[\mathbf{h}_p^2]$ (with $\mathcal{M}_p$ replaced by identity) |
| `H_1_d` | Diagonal of first Hessian moment: $H_{ii} = \mathbf{v}_i^\top \mathbb{E}_p[\mathbf{h}_p] \mathbf{v}_i$ |
| `H_2_d` | Diagonal of second Hessian moment: $\mathbf{v}_i^\top \mathbb{E}_p[\mathbf{h}_p^2] \mathbf{v}_i$ |
| `Covar` | Empirical noise covariance via Eq. 2: $\mathbf{C} = \frac{1}{B}[\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^\top - \mathbf{g}\mathbf{g}^\top]$ |
| `Hessian` | Global Hessian $\mathbf{H} = \nabla^2 \mathcal{L}$ and its eigen-decomposition (`components`) |

All matrices are computed and stored in the **Hessian eigenbasis** $\{\mathbf{v}_i\}$ (referred to as `components` in the code).

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

## Experimental Setup

All models are trained with vanilla SGD (no extra regularization) to convergence: **100% training accuracy** for CE loss, or **>95%** for MSE loss. A *Softmax* layer is applied before MSE to stabilize Hessian spectra.

### Architecture Details

**MLP (MNIST):** Two hidden layers of width 50 ($784 \to 50 \to 50 \to 10$), ReLU activations, no bias. The AWD analysis targets the weight matrix between the two hidden layers (flattened parameter dimension $D = 50 \times 50 = 2500$).

**MLP (CIFAR-10):** Three hidden layers ($3072 \to 1000 \to 50 \to 50 \to 10$), ReLU activations, no bias. The AWD analysis targets the weight matrix connecting the last two hidden layers ($D = 50 \times 50 = 2500$).

**CNN (MNIST & CIFAR-10):** VGG-style convolutional layers `[32, M, 64, M, 128, 128, M]` (kernel size 3, padding 1; `M` = MaxPool with kernel 2, stride 2), followed by AdaptiveAvgPool → $128 \to 20 \to \mathcal{C}$ fully connected classifier. ReLU activations, no bias, no BatchNorm. The AWD analysis targets the weight matrix from features to the hidden FC layer ($D = 128 \times 20 = 2560$).

All intermediate layer features are cached in `self.feature` for per-sample Hessian computation.

| Model | `layer_index` | Parameter Dimension $D$ |
|-------|---------------|------------------------|
| MLP (MNIST) | [1] | 2500 |
| MLP (CIFAR-10) | [1] | 2500 |
| CNN | [8] | 2560 |

### Training Hyperparameters (Table 1 in paper)

The following table specifies the exact training setups used to produce the main results. $N_{\text{data}}$ denotes samples per class; $\mathcal{N}$ is the number of top eigenvalues used for $\gamma$ fitting; $\mathcal{C}$ is the number of classes.

| Dataset | Model | Loss | $N_{\text{data}}$ | Batch $B$ | Epochs | $\mathcal{N}$ ($\mathcal{C}$=3) | $\mathcal{N}$ ($\mathcal{C}$=6) | $\mathcal{N}$ ($\mathcal{C}$=10) |
|---------|-------|------|----------|---------|--------|------|------|------|
| MNIST | MLP | CE | 2,000 | 50 | 100 | 300 | 1,000 | 1,000 |
| MNIST | MLP | MSE | 2,000 | 50 | 100 | 300 | 1,000 | 1,000 |
| MNIST | CNN | CE | 2,000 | 50 | 100 | 200 | 500 | 1,000 |
| MNIST | CNN | MSE | 5,000 | 128 | 100 | 200 | 300 | 800 |
| CIFAR-10 | MLP | CE | 2,000 | 100 | 150 | 800 | 1,500 | 1,500 |
| CIFAR-10 | MLP | MSE | 5,000 | 100 | 100 | 500 | 1,000 | 1,000 |
| CIFAR-10 | CNN | CE | 2,000* | 128 | 100 | 500 | 1,000 | 1,500 |
| CIFAR-10 | CNN | MSE | 5,000 | 128 | 500 | 500 | 500 | 1,000 |

\* For CIFAR-10 CNN CE with $\mathcal{C}=3$, $N_{\text{data}}=5{,}000$.

All experiments use SGD with learning rate $\eta = 0.1$. Results in Table 1 are averaged over **4 independent runs** with distinct random seeds.

### Figure-Specific Settings

- **CNN figures:** Trained on a balanced CIFAR-10 subset (2,000 per class, 20,000 total). CE loss, 100 epochs, $B=128$, $\eta=0.1$.
- **MLP figures:** Trained on a balanced MNIST subset (2,000 per class, 20,000 total). 100 epochs, $B=50$, $\eta=0.1$.

### Configuration Parameters (model_config.py)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.1 | SGD learning rate $\eta$ |
| `lss_fn` | str | `'mse'` | Loss function: `'mse'` (Softmax+MSE) / `'cse'` (CrossEntropy) / `'lmse'` (Linear MSE) |
| `B` | int | 50 | Mini-batch size |
| `train_size` | int | 2000 | Training samples per class $N_{\text{data}}$ |
| `test_size` | int | 1000 | Test set size |
| `rho` | float | 0 | Label noise probability |
| `net_size` | int | 50 | Hidden layer width |
| `s` | int | 1 | Weight initialization scaling factor |
| `d` | float | 0 | Dropout probability |
| `beta` | float | 0 | L2 regularization coefficient |
| `stop_loss` | float | 1e-5 | Early stopping loss threshold |
| `sample_holder` | list | [0..9] | Class IDs for matched sample pair construction |
| `class_number` | int | 10 | Total number of classes $\mathcal{C}$ |
| `layer_index` | list | [1] | Target layer index for AWD analysis |
| `dataset` | str | — | Dataset: `'mnist'` / `'cifar10'` / `'fdata'` |
| `model` | str | — | Architecture: `'FC'` / `'MLP'` / `'CNN'` |

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| MNIST | Handwritten digits, 28×28 grayscale |
| CIFAR-10 | 10-class natural images, 32×32 color (normalized) |

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
@article{zhang2026superlinear,
  title={On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature},
  author={Zhang, Yikuan and Yang, Ning and Tu, Yuhai},
  year={2026}
}
```

# AW-CH: Approximate Weight Duality — Covariance & Hessian Analysis

A research framework for analyzing neural network generalization through loss landscape curvature (Hessian) and SGD noise covariance, based on **Approximate Weight Duality (AWD)**.

## Core Ideas

1. **AWD (Approximate Weight Duality)**: Maps input-space perturbations ($\Delta x$) between samples to equivalent weight-space perturbations ($w$), establishing a duality:

$$w_{ij} = \frac{\sum_k W_{ik} \Delta x_k \cdot x_j}{\|x\|^2}$$

2. **Per-sample Hessian Analysis**: Computes the per-sample Hessian $H_s$ of the loss w.r.t. a target layer's parameters, and studies its statistics ($\mathbb{E}[H]$, $\mathbb{E}[H^2]$).

3. **C Matrix Decomposition**: Computes three matrices $C_1$, $C_2$, $C_3$ characterizing interactions among Hessian, weight perturbations ($w$), gradients ($g$), and input perturbation operators ($a$):
   - $C_1 = \mathbb{E}[H w w^T H^T]$ — Hessian–weight-perturbation second moment
   - $C_2 = \mathbb{E}[H w g^T a]$ — Cross term
   - $C_3 = \mathbb{E}[g a a^T g^T]$ — Gradient–input-perturbation term
   - Each $C_i$ is computed in both **diagonal** (same-sample pairs) and **full** (cross-sample pairs) forms.

4. **Noise Covariance**: Computes the SGD noise covariance $C = F - \nabla L \nabla L^T$, where $F$ is the Fisher information matrix.

All computations are performed in the Hessian eigenbasis (`components`), i.e., $H' = P H P^T$ where $P$ contains the global Hessian's eigenvectors.

## Project Structure

```
├── model_config.py          # Hyperparameter configuration and sweep ranges
├── data.py                  # Data loading (MNIST / CIFAR-10), subset sampling, label noise injection
├── models.py                # Model definitions (FC / MLP / CNN) with intermediate feature caching
├── train_model.py           # Training loop: SGD + CosineAnnealingLR, gradient clipping, early stopping
├── utils.py                 # Hessian, gradient, Fisher information, noise covariance computation
├── AWD_cuda.py              # Core AWD algorithm (CUDA): C matrix computation via torch.func vmap/hessian/grad
│
├── cal_C_cuda_multi.py      # Entry point: multi-run C matrix computation (different random seeds)
├── cal_h_g.py               # Entry point: per-sample Hessian & gradient computation
│
├── Figures.ipynb            # Visualization: C-H commutativity analysis, log-log plots
└── effective_power.ipynb    # Effective power analysis notebook
```

## Pipeline

```
python cal_C_cuda_multi.py --max_e 100 --net_size 50 --n_class 10 --runs 5
  │
  ├─ 1. train_model.train()            # Train model (SGD + CosineAnnealing)
  ├─ 2. utils.cal_hessian_cuda()       # Global Hessian → eigen-decomposition → components
  ├─ 3. utils.cal_noise_covar_*()      # SGD noise covariance
  ├─ 4. AWD_cuda.cal_C_cuda()          # Compute C1, C2, C3 (diagonal + full)
  └─ 5. torch.save()                   # Save results to AWCH_data/
```

For per-sample Hessian & gradient only:
```
python cal_h_g.py --max_e 100
```

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.1 | SGD learning rate |
| `lss_fn` | str | `'mse'` | Loss function: `'mse'` (Softmax+MSE) / `'cse'` (CrossEntropy) / `'lmse'` (Linear MSE) |
| `B` | int | 50 | Training batch size |
| `train_size` | int | 2000 | Training samples per class |
| `test_size` | int | 1000 | Test set size |
| `rho` | float | 0 | Label noise probability |
| `net_size` | int | 50 | Hidden layer width |
| `s` | int | 1 | Weight initialization scaling factor |
| `d` | float | 0 | Dropout probability |
| `beta` | float | 0 | L2 regularization coefficient |
| `stop_loss` | float | 1e-5 | Early stopping loss threshold |
| `sample_holder` | list | [0..9] | Class IDs for AWCH computation |
| `class_number` | int | 10 | Total number of classes |
| `layer_index` | list | [1] | Target layer index (FC/MLP: [1]; CNN: [8]) |
| `dataset` | str | — | Dataset: `'mnist'` / `'cifar10'` / `'fdata'` |
| `model` | str | — | Architecture: `'FC'` / `'MLP'` / `'CNN'` |

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
  ├── C_epoch_{epoch}_run_{id}.pt        # C1, C2, C3 matrices (diagonal & full), Hessian stats, covariance
  └── h_g_tensors_holder_epoch_{epoch}.pt  # Per-sample Hessian & gradient tensors, components
```

Each `.pt` file contains: `C1_dia`, `C1_dia_w_dia`, `C1_h`, `C2_dia`, `C3_dia`, `C1`, `C2`, `C3`, `C` (noise covariance), `H_1_d`, `H_2_d`, `Covar`, `Hessian`, training/test loss and accuracy.

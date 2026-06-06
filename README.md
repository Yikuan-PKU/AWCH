# AW-CH: On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature

Code for the paper: *"On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature"*.

## Overview

This repository provides the implementation for studying the relationship between the SGD noise covariance $\mathbf{C}$ and the loss Hessian $\mathbf{H}$ in neural networks, based on the **Activity-Weight Duality (AWD)** framework.

**Key findings:**
- The noise covariance is governed by the **second moment** of per-sample Hessians: $\mathbf{C} \propto \mathbb{E}_p[\mathbf{h}_p^2]$, where $\mathbf{H} = \mathbb{E}_p[\mathbf{h}_p]$.
- $\mathbf{C}$ and $\mathbf{H}$ approximately commute ($[\mathbf{C}, \mathbf{H}] \approx 0$) rather than coincide.
- Their diagonal elements follow a **power-law** relation $C_{ii} \propto H_{ii}^{\gamma}$ with a theoretically bounded exponent $1 \leq \gamma \leq 2$.
- Cross-entropy (CE) loss exhibits superlinear scaling ($\gamma > 1$, up to about 1.4), while MSE loss yields approximately linear scaling ($\gamma \approx 1$).

## Method

### Activity-Weight Duality (AWD)

For a fully connected layer with weights $\mathbf{W}$, given an input activity perturbation $\Delta \mathbf{a}$ from a matched sample pair, the **Minimal AWD** finds the weight perturbation $\Delta \mathbf{W}^*$ that preserves pre-activations with minimal Frobenius norm:

$$\Delta \mathbf{W}^* = \frac{(\mathbf{W} \Delta \mathbf{a}) \mathbf{a}^\top}{\|\mathbf{a}\|^2}$$

### AWD-Based Noise Covariance Decomposition

Under the AWD gradient approximation, the gradient difference between two mini-batches is dominated by the Hessian-driven term near convergence:

$$\mathbf{g}_{\mu\nu} \approx \frac{1}{B} \sum_{p \in \mathcal{B}_\nu} \mathbf{h}_p(\mathbf{w}) \Delta \mathbf{w}_p^{\mu\nu}$$

This yields the noise covariance as a quadratic form of the per-sample Hessian (Theorem 1 in the paper):

$$C_{ij} \approx \frac{\sigma_w^2}{2B} \mathbb{E}_p \left[ \sum_m (\kappa_m^{(p)})^2 (\mathbf{u}_m^{(p)} \cdot \mathbf{v}_i)(\mathbf{u}_m^{(p)} \cdot \mathbf{v}_j) \right]$$

where $\kappa_m^{(p)}$ and $\mathbf{u}_m^{(p)}$ are the eigenvalues and eigenvectors of the per-sample Hessian $\mathbf{h}_p$, and $\{\mathbf{v}_i\}$ is the global Hessian eigenbasis.

### Code Decomposition

In the code, the AWD-based covariance is decomposed into three components corresponding to the full expansion:

| Code Variable | Paper Notation | Formula |
|---------------|----------------|---------|
| `C1` / `C1_dia` | $\mathbf{C}^{hh}$ | $\mathbb{E}[\mathbf{h}_p \Delta\mathbf{w}_p \Delta\mathbf{w}_p^\top \mathbf{h}_p^\top]$, the pure Hessian-weight contribution |
| `C2` / `C2_dia` | $\mathbf{C}^{hg}$ | Cross-interaction between Hessian-weight and gradient-activity terms |
| `C3` / `C3_dia` | $\mathbf{C}^{gg}$ | Pure gradient-activity contribution |

- `*_dia` variants: same-sample contributions in the sample-summation sense.
- Without `_dia`: full terms including cross-sample contributions.

Additional stored quantities:

| Code Variable | Description |
|---------------|-------------|
| `C1_dia_w_dia` | Same-sample Hessian term with only the diagonal of the local perturbation covariance $\mathcal{M}_p$ retained |
| `C1_h` | Hessian second moment $\mathbb{E}_p[\mathbf{h}_p^2]$ with $\mathcal{M}_p$ replaced by identity |
| `H_1_d` | First Hessian moment $\mathbb{E}_p[\mathbf{h}_p]$ represented in the Hessian eigenbasis |
| `H_2_d` | Second Hessian moment $\mathbb{E}_p[\mathbf{h}_p^2]$ represented in the Hessian eigenbasis |
| `Covar` | Empirical noise covariance via Eq. 2 |
| `Hessian` | Global Hessian $\mathbf{H} = \nabla^2 \mathcal{L}$ |

## Project Structure

```
model_config.py       # Hyperparameter configuration and sweep ranges
data.py               # Data loading, subset sampling, and preprocessing
models.py             # FC, FC_multilayer, MLP, and CNN model definitions
train_model.py        # Training with SGD, CosineAnnealingLR, clipping, and early stopping
utils.py              # Hessian, Fisher information, and noise covariance utilities
AWD_cuda.py           # Core AWD and per-sample Hessian computations
cal_C_cuda.py         # Entry point: single-layer AWD covariance bundle
cal_H1_H2.py          # Entry point: joint single-layer or multilayer H1/H2 statistics
cal_h_g.py            # Entry point: per-sample Hessian h_p and gradient export
commutativity.py      # Commutativity and eigenbasis-alignment analysis
Figures.ipynb         # Visualization: C-H commutativity and log-log power-law plots
powerLawPlot.ipynb    # Power-law plotting utilities
Ablation_exp.ipynb    # Suppression experiment analysis
```

## Pipeline

### AWD covariance bundle

```bash
python cal_C_cuda.py --max_e 100 --net_size 50 --n_class 10 --layer_index 1
```

This path runs:

```
train_model.train()
  -> utils.cal_hessian_cuda()
  -> utils.cal_noise_covar_minibatch()
  -> AWD_cuda.cal_C_cuda()
  -> torch.save()
```

`cal_C_cuda.py` currently uses `FC_multilayer` with `hidden_sizes = [50, 40, 30, 20]` and computes AWD-based `C1/C2/C3` for one selected layer per run.

### Joint H1/H2 statistics

```bash
python cal_H1_H2.py --max_e 200 --layer_indices "1,2,3"
```

This path runs:

```
train_model.train()
  -> utils.cal_hessian_cuda()
  -> utils.cal_noise_covar_minibatch()
  -> AWD_cuda.cal_hessian_stats_cuda_multi()
  -> torch.save()
```

`cal_H1_H2.py` supports comma-separated `--layer_indices`. When multiple layers are passed, it builds a joint target parameter space and computes `H_1_d` and `H_2_d` in the corresponding Hessian eigenbasis.

### Per-sample Hessian and gradient export

```bash
python cal_h_g.py --max_e 100
```

### Commutativity analysis

```bash
python commutativity.py
```

`commutativity.py` analyzes saved tensors and reports power-law, rank-correlation, commutator, top/bulk projected commutator, diagonal-reconstruction, and spectrum-preserving random-baseline diagnostics. The script uses editable configuration values near the `if __name__ == "__main__":` block.

## Reproducing Paper Results

The key experiments in the paper can be reproduced as follows:

| Paper Content | Code Entry |
|---------------|------------|
| Table 1 ($\gamma_\text{emp}$ vs $\gamma_\text{AWD}$) | `cal_C_cuda.py` with the desired dataset, loss, class count, and layer setting |
| Joint multilayer H1/H2 statistics | `cal_H1_H2.py --layer_indices "1,2,3"` |
| Fig. 1 (C-H commutativity) | `Figures.ipynb` and `commutativity.py` |
| Fig. 3 (Log-log power law) | `Figures.ipynb` and `powerLawPlot.ipynb` |
| Fig. 5 (Suppression experiment) | `Ablation_exp.ipynb` |
| Per-sample Hessian $\mathbf{h}_p$ statistics | `cal_h_g.py` |

## Command-Line Arguments

### cal_C_cuda.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_e` | int | 200 | Maximum number of training epochs |
| `--net_size` | int | 50 | Hidden layer width |
| `--n_class` | int | 10 | Number of classification classes |
| `--layer_index` | int | 1 | Target layer used for Hessian and AWD covariance calculations |

### cal_H1_H2.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_e` | int | 200 | Maximum number of training epochs |
| `--layer_indices` | str | `"1"` | Comma-separated target layers, for example `"1"` or `"1,2,3"` |

### cal_h_g.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_e` | int | 200 | Maximum number of training epochs |

## Experimental Setup

All models are trained with vanilla SGD to convergence: **100% training accuracy** for CE loss, or **>95%** for MSE loss. A Softmax layer is applied before MSE to stabilize Hessian spectra.

### Architecture Details

**MLP (MNIST):** Two hidden layers of width 50 ($784 \to 50 \to 50 \to 10$), ReLU activations, no bias. The AWD analysis targets the weight matrix between the two hidden layers.

**MLP (CIFAR-10):** Three hidden layers ($3072 \to 1000 \to 50 \to 50 \to 10$), ReLU activations, no bias. The AWD analysis targets the weight matrix connecting the last two hidden layers.

**FC_multilayer (MNIST):** A fully connected model with configurable hidden sizes. The current AWD covariance path uses `[50, 40, 30, 20]`, while the joint H1/H2 path uses `[50, 50, 50, 50]`.

**CNN (MNIST & CIFAR-10):** VGG-style convolutional layers followed by a fully connected classifier. The AWD analysis targets the feature-to-hidden fully connected weight matrix.

All intermediate layer features are cached in `self.feature` for per-sample Hessian computation.

| Model | Typical `layer_index` / `layer_indices` | Notes |
|-------|-----------------------------------------|-------|
| MLP (MNIST) | `[1]` | Single-layer AWD analysis |
| MLP (CIFAR-10) | `[2]` | Single-layer AWD analysis |
| CNN | `[8]` | Feature-to-hidden classifier layer |
| FC_multilayer | `[1]`, `[1,2,3]`, or `[1,2,3,4]` | Single-layer AWD C or joint H1/H2 statistics |

### Training Hyperparameters (Table 1 in paper)

The following table specifies the training setups used to produce the main results. $N_{\text{data}}$ denotes samples per class; $\mathcal{N}$ is the number of top eigenvalues used for $\gamma$ fitting; $\mathcal{C}$ is the number of classes.

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

\* For CIFAR-10 CNN CE with $\mathcal{C}=3$, $N_{\text{data}}=5,000$.

All experiments use SGD with learning rate $\eta = 0.1$. Results in Table 1 are averaged over **4 independent runs** with distinct random seeds.

### Figure-Specific Settings

- **CNN figures:** Trained on a balanced CIFAR-10 subset (2,000 per class, 20,000 total). CE loss, 100 epochs, $B=128$, $\eta=0.1$.
- **MLP figures:** Trained on a balanced MNIST subset (2,000 per class, 20,000 total). 100 epochs, $B=50$, $\eta=0.1$.

### Configuration Parameters (model_config.py)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.1 | SGD learning rate $\eta$ |
| `lss_fn` | str | `'mse'` | Loss function: `'mse'`, `'cse'`, or `'lmse'` |
| `B` | int | 50 | Mini-batch size |
| `train_size` | int | 2000 | Training samples per class |
| `test_size` | int | 1000 | Test set size |
| `rho` | float | 0 | Label noise probability |
| `net_size` | int | 50 | Hidden layer width |
| `s` | int | 1 | Weight initialization scaling factor |
| `d` | float | 0 | Dropout probability |
| `beta` | float | 0 | L2 regularization coefficient |
| `stop_loss` | float | 1e-5 | Early stopping loss threshold |
| `sample_holder` | list | `[0..9]` | Class IDs for matched sample pair construction |
| `class_number` | int | 10 | Total number of classes |
| `layer_index` | list | `[1]` | Target layer or target layer list |
| `dataset` | str | `-` | Dataset: `'mnist'` or `'cifar10'` |
| `model` | str | `-` | Architecture: `'FC'`, `'FC_multilayer'`, `'MLP'`, or `'CNN'` |

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| MNIST | Handwritten digits, 28x28 grayscale |
| CIFAR-10 | 10-class natural images, 32x32 color |

## Getting Started

### Requirements

- Python 3.8+
- PyTorch with CUDA support
- torchvision
- numpy, matplotlib, scipy

### Installation

```bash
pip install torch torchvision numpy matplotlib scipy
```

### Run

```bash
# Compute the AWD covariance bundle for one layer
python cal_C_cuda.py --max_e 100 --net_size 50 --n_class 10 --layer_index 1

# Compute joint H1/H2 statistics for multiple layers
python cal_H1_H2.py --max_e 200 --layer_indices "1,2,3"

# Compute per-sample Hessian and gradients
python cal_h_g.py --max_e 100

# Run commutativity and eigenbasis-alignment analysis
python commutativity.py
```

## Output Data Format

Results are saved under `AWCH_data/`.

The AWD covariance bundle is saved as:

```
AWCH_data/HS{hidden_sizes}_layer{layer_index}_TrainSize{train_size}_SampleN{sample_number}_ClassN{n_class}_B{batch}lr{lr}_lossfn_{loss}_model_{model}_dataset_{dataset}/
  C_epoch_{epoch}.pt
```

The H1/H2 bundle is saved as:

```
AWCH_data/HS{hidden_sizes}_layer{layer_indices}_TrainSize{train_size}_SampleN{sample_number}_ClassN{n_class}_B{batch}lr{lr}_lossfn_{loss}_model_{model}_dataset_{dataset}/
  H1_H2_epoch_{epoch}.pt
```

**Contents of `C_epoch_*.pt`:**

| Key | Description | Paper Reference |
|-----|-------------|-----------------|
| `C1_dia`, `C1` | $\mathbf{C}^{hh}$: Hessian-weight contribution | Term I x Term I |
| `C2_dia`, `C2` | $\mathbf{C}^{hg}$: Cross-interaction | Term I x Term II |
| `C3_dia`, `C3` | $\mathbf{C}^{gg}$: Gradient-activity contribution | Term II x Term II |
| `C1_dia_w_dia` | $\mathbf{C}^{hh}$ with diagonal-only local perturbation covariance | Isotropy check |
| `C1_h` | $\mathbb{E}_p[\mathbf{h}_p^2]$ with $\mathcal{M}_p = \mathbf{I}$ | Core second-moment result |
| `H_1_d` | First Hessian moment in Hessian eigencoordinates | Diagonal gives $H_{ii}$ |
| `H_2_d` | Second Hessian moment in Hessian eigencoordinates | Diagonal gives the second-moment quantity |
| `C` | AWD reconstruction `C1 + C2 + C3` | AWD covariance approximation |
| `Covar` | Empirical noise covariance | Eq. 2 |
| `Hessian` | Global Hessian $\mathbf{H}$ | Hessian basis source |

**Contents of `H1_H2_epoch_*.pt`:**

| Key | Description |
|-----|-------------|
| `H_1_d` | First Hessian moment in Hessian eigencoordinates |
| `H_2_d` | Second Hessian moment in Hessian eigencoordinates |
| `layer_index` | Target layer or joint target layer list |
| `Covar` | Empirical noise covariance |
| `Hessian` | Global Hessian |
| metric histories | Train/test loss and accuracy histories |

## Basis Conventions

- `Hessian` and `Covar` are saved in the original parameter basis.
- `C1`, `C2`, `C3`, `C1_h`, `C1_dia_w_dia`, `H_1_d`, and `H_2_d` are saved in Hessian eigencoordinates.
- `H_1_d` and `H_2_d` are saved as full matrices; downstream analysis usually takes their diagonals explicitly.
- `components` is not saved in the main bundles and can be recomputed from `Hessian` when needed.

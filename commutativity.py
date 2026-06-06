import torch
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import os
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

def set_publication_style():
    """Implementation note."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 2,
        'lines.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
set_publication_style()
# --------------------------
# --------------------------
def extract_diagonals(A, B, slice_range=None):
    if slice_range is None:
        slice_range = torch.arange(min(A.shape[0], B.shape[0]))
    A = torch.flip(A, dims=[0,1])
    B = torch.flip(B, dims=[0,1])
    A_dense = torch.diag_embed(torch.diag(A)[slice_range])
    B_dense = torch.diag_embed(torch.diag(B)[slice_range])
    return A_dense, B_dense

def compute_loglog_fit(A_dense, B_dense):
    diag_a = A_dense.diag().detach().cpu().numpy()
    diag_b = B_dense.diag().detach().cpu().numpy()
    mask = (diag_a > 0) & (diag_b > 0)
    if mask.sum() == 0:
        return None, None, None, None
    log_a, log_b = np.log10(diag_a[mask]), np.log10(diag_b[mask])
    slope, intercept, r_value, _, _ = stats.linregress(log_b, log_a)
    return log_b, log_a, slope, r_value**2

def compute_spearman_rank(A_dense, B_dense):
    diag_a = A_dense.diag().detach().cpu().numpy()
    diag_b = B_dense.diag().detach().cpu().numpy()
    # define mask for idx>2000
    # mask = np.arange(len(diag_a)) > 1000
    mask = (diag_a > 0 ) & (diag_b > 0)
    if mask.sum() < 2: return 0.0
    correlation, _ = stats.spearmanr(diag_a[mask], diag_b[mask])
    return correlation



def compute_commutativity_random_baseline(A, B, k=100, m=1000):
    """Spectrum-preserving random baseline for full/top/bulk commutativity."""
    if A.ndim == 1: A = torch.diag(A)
    if B.ndim == 1: B = torch.diag(B)
    A, B = A.float().cpu(), B.float().cpu()
    if A.shape != B.shape:
        return None, None, None
    try:
        A_rand = _spectrum_preserving_random_matrix(A)
        B_rand = _spectrum_preserving_random_matrix(B)
        if A_rand is None or B_rand is None:
            return None, None, None
        comm_rand = compute_commutativity(A_rand, B_rand)
        top_rand, bulk_rand = compute_split_commutativity(A_rand, B_rand, k=k, m=m)
        return comm_rand, top_rand, bulk_rand
    except Exception as e:
        print(f"Error in commutativity random baseline: {e}")
        return None, None, None

    """
    """
    if A.ndim == 1: A = torch.diag(A)
    if B.ndim == 1: B = torch.diag(B)
    A, B = A.float().cpu(), B.float().cpu()
    N = B.shape[0]
    if k >= N: k = max(1, N // 10)
    if m > N: m = N
    if m <= k: m = min(N, k + 100)
    return None, None, None

    N = B.shape[0]
    H = torch.randn(N, N, device=B.device)
    Q, _ = torch.linalg.qr(H)
    
    B_rand = Q @ B @ Q.T
    
    return compute_commutativity(A, B_rand)

def compute_commutativity(A, B):
    """Standard Frobenius-normalized commutator: ||AB - BA||_F / (||A||_F ||B||_F)."""
    if A.ndim == 1: A = torch.diag(A)
    if B.ndim == 1: B = torch.diag(B)
    A, B = A.float().cpu(), B.float().cpu()
    if A.shape != B.shape: return None 
    AB = torch.matmul(A, B)
    BA = torch.matmul(B, A)
    commutator = AB - BA
    diff_norm = torch.norm(commutator, p='fro')
    base_norm = torch.norm(A, p='fro') * torch.norm(B, p='fro')
    if base_norm == 0: return 0.0
    return (diff_norm / base_norm).item()

def compute_eigen_alignment(target_A, basis_B):
    """Per-entry offdiag/diag RMS ratio of A in B's eigenbasis; smaller is more diagonal."""
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    A, B = target_A.float().cpu(), basis_B.float().cpu()
    try:
        L_B, V_B = torch.linalg.eigh(B) 
    except:
        return None 
    A_rotated = V_B.T @ A @ V_B
    diag = torch.diag(A_rotated)
    diag_rms = torch.sqrt(torch.mean(diag ** 2))
    if diag_rms == 0: return 0.0
    off_diag = A_rotated - torch.diag_embed(diag)
    n = A_rotated.shape[0]
    if n <= 1: return 0.0
    offdiag_rms = torch.sqrt(torch.sum(off_diag ** 2) / (n * (n - 1)))
    return (offdiag_rms / diag_rms).item()

def _spectrum_preserving_random_matrix(A):
    """Randomize eigenvectors while preserving A's eigenvalue spectrum."""
    if A.ndim == 1:
        A = torch.diag(A)
    A = A.float().cpu()
    if A.shape[0] == 0:
        return None
    A_sym = 0.5 * (A + A.T)
    eigvals = torch.linalg.eigvalsh(A_sym)
    Q, _ = torch.linalg.qr(torch.randn(A.shape[0], A.shape[0], device=A.device, dtype=A.dtype))
    return Q @ torch.diag(eigvals) @ Q.T

def compute_split_eigen_alignment(target_A, basis_B, k=100, m=1000):
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    A, B = target_A.float().cpu(), basis_B.float().cpu()

    try:
        L_B, V_B = torch.linalg.eigh(B)
        N = V_B.shape[0]
        if k >= N: k = max(1, N // 10)
        if m > N: m = N
        if m <= k: m = min(N, k + 100)

        def leakage(V_sub):
            K = V_sub.T @ A @ V_sub
            diag = torch.diag(K)
            diag_rms = torch.sqrt(torch.mean(diag ** 2))
            if diag_rms == 0: return 0.0
            off = K - torch.diag_embed(diag)
            n = K.shape[0]
            if n <= 1: return 0.0
            offdiag_rms = torch.sqrt(torch.sum(off ** 2) / (n * (n - 1)))
            return (offdiag_rms / diag_rms).item()

        return leakage(V_B[:, -k:]), leakage(V_B[:, -m:-k])
    except Exception as e:
        print(f"Error in split eigen alignment: {e}")
        return None, None

def compute_eigen_alignment_random_baseline(A, B, k=100, m=1000):
    if A.ndim == 1: A = torch.diag(A)
    if B.ndim == 1: B = torch.diag(B)
    A, B = A.float().cpu(), B.float().cpu()
    if A.shape != B.shape:
        return None, None, None
    N = B.shape[0]
    if k >= N: k = max(1, N // 10)
    if m > N: m = N
    if m <= k: m = min(N, k + 100)
    try:
        A_rand = _spectrum_preserving_random_matrix(A)
        if A_rand is None:
            return None, None, None
        rand_full = compute_eigen_alignment(A_rand, B)
        rand_top, rand_bulk = compute_split_eigen_alignment(A_rand, B, k=k, m=m)
        return rand_full, rand_top, rand_bulk
    except Exception as e:
        print(f"Error in eigen alignment random baseline: {e}")
        return None, None, None

def _diagonal_reconstruction_recovery_from_block(K):
    """Fraction of Frobenius energy recovered by keeping only diag(K)."""
    if K.numel() == 0 or K.shape[0] == 0:
        return None
    total_energy = torch.sum(K ** 2)
    if total_energy == 0:
        return 0.0
    diag_energy = torch.sum(torch.diag(K) ** 2)
    return torch.clamp(diag_energy / total_energy, 0.0, 1.0).item()

def compute_diagonal_reconstruction_recovery(target_A, basis_B):
    """Energy fraction of A recovered by its diagonal part in B's eigenbasis."""
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    A, B = target_A.float().cpu(), basis_B.float().cpu()
    if A.shape != B.shape:
        return None
    try:
        _, V_B = torch.linalg.eigh(B)
        K = V_B.T @ A @ V_B
        return _diagonal_reconstruction_recovery_from_block(K)
    except Exception as e:
        print(f"Error in diagonal reconstruction recovery: {e}")
        return None

def compute_split_diagonal_reconstruction_recovery(target_A, basis_B, k=100, m=1000):
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    A, B = target_A.float().cpu(), basis_B.float().cpu()
    if A.shape != B.shape:
        return None, None
    try:
        _, V_B = torch.linalg.eigh(B)
        N = V_B.shape[0]
        if k >= N: k = max(1, N // 10)
        if m > N: m = N
        if m <= k: m = min(N, k + 100)

        V_top = V_B[:, -k:]
        K_top = V_top.T @ A @ V_top
        top_recovery = _diagonal_reconstruction_recovery_from_block(K_top)

        bulk_recovery = None
        if m > k:
            V_bulk = V_B[:, -m:-k]
            K_bulk = V_bulk.T @ A @ V_bulk
            bulk_recovery = _diagonal_reconstruction_recovery_from_block(K_bulk)

        return top_recovery, bulk_recovery
    except Exception as e:
        print(f"Error in split diagonal reconstruction recovery: {e}")
        return None, None

def compute_reconstruction_random_baseline(A, B, k=100, m=1000):
    """Spectrum-preserving random baseline for diagonal reconstruction recovery."""
    if A.ndim == 1: A = torch.diag(A)
    if B.ndim == 1: B = torch.diag(B)
    A, B = A.float().cpu(), B.float().cpu()
    if A.shape != B.shape:
        return None, None, None

    N = A.shape[0]
    if k >= N: k = max(1, N // 10)
    if m > N: m = N
    if m <= k: m = min(N, k + 100)

    try:
        A_sym = 0.5 * (A + A.T)
        eigvals = torch.linalg.eigvalsh(A_sym)
        Q, _ = torch.linalg.qr(torch.randn(N, N, device=A.device, dtype=A.dtype))
        K_rand = Q @ torch.diag(eigvals) @ Q.T

        full_recovery = _diagonal_reconstruction_recovery_from_block(K_rand)
        top_recovery = _diagonal_reconstruction_recovery_from_block(K_rand[-k:, -k:])
        bulk_recovery = None
        if m > k:
            bulk_recovery = _diagonal_reconstruction_recovery_from_block(K_rand[-m:-k, -m:-k])

        return full_recovery, top_recovery, bulk_recovery
    except Exception as e:
        print(f"Error in reconstruction random baseline: {e}")
        return None, None, None

# def compute_split_commutativity(target_A, basis_B, top_k=100):
#     if target_A.ndim == 1: target_A = torch.diag(target_A)
#     if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
#     A, B = target_A.float().cpu(), basis_B.float().cpu()
#     try:
#         L_B, V_B = torch.linalg.eigh(B)
#         V_top = V_B[:, -top_k:]  
#         P_top = V_top @ V_top.T
#         I = torch.eye(B.shape[0], device=B.device)
#         P_bulk = I - P_top
#         Comm = A @ B - B @ A
#         AB = A @ B 
#         num_top = torch.norm(P_top @ Comm @ P_top, p='fro')
#         den_top = torch.norm(P_top @ AB @ P_top, p='fro')
#         err_top = (num_top / (den_top )).item()
#         num_bulk = torch.norm(P_bulk @ Comm @ P_bulk, p='fro')
#         den_bulk = torch.norm(P_bulk @ AB @ P_bulk, p='fro')
#         err_bulk = (num_bulk / (den_bulk )).item()
#         return err_top, err_bulk
#     except Exception as e:
#         return None, None

def compute_split_commutativity(target_A, basis_B, k=100, m=1500):

    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    A, B = target_A.float().cpu(), basis_B.float().cpu()
    
    try:
        L_B, V_B = torch.linalg.eigh(B)
        N = V_B.shape[0]
        
        if k >= N: k = N // 10
        if m > N: m = N
        if m <= k: m = k + 100
        
        # Indices: [N-k, N]
        V_top = V_B[:, -k:]
        
        # Indices: [N-m, N-k]
        V_bulk = V_B[:, -m:-k]
        
        P_top = V_top @ V_top.T
        P_bulk = V_bulk @ V_bulk.T
        
        Comm = A @ B - B @ A
        
        num_top = torch.norm(P_top @ Comm @ P_top, p='fro')
        A_top = P_top @ A @ P_top
        B_top = P_top @ B @ P_top
        den_top = torch.norm(A_top, p='fro') * torch.norm(B_top, p='fro')
        err_top = 0.0 if den_top == 0 else (num_top / den_top).item()
        
        num_bulk = torch.norm(P_bulk @ Comm @ P_bulk, p='fro')
        A_bulk = P_bulk @ A @ P_bulk
        B_bulk = P_bulk @ B @ P_bulk
        den_bulk = torch.norm(A_bulk, p='fro') * torch.norm(B_bulk, p='fro')
        err_bulk = 0.0 if den_bulk == 0 else (num_bulk / den_bulk).item()
        
        return err_top, err_bulk
        
    except Exception as e:
        print(f"Error in windowed commutativity: {e}")
        return None, None



# def compute_scale_invariant_stats(target_A, basis_B):
#     if target_A.ndim == 1: target_A = torch.diag(target_A)
#     if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
#     A, B = target_A.float().cpu(), basis_B.float().cpu()
#     try:
#         L_B, V_B = torch.linalg.eigh(B)
#         idx_flip = torch.arange(L_B.shape[0] - 1, -1, -1)
#         V_B = V_B[:, idx_flip]
#         L_B = L_B[idx_flip]
        
#         # K: Raw projection
#         K = V_B.T @ A @ V_B
        
#         # R: Normalized correlation (signed)
#         diag_val = torch.abs(torch.diagonal(K))
#         norm_factor = torch.sqrt(torch.outer(diag_val, diag_val))
#         R = K / norm_factor 
#         R_abs = torch.abs(R) 

#         # Stats based on absolute value off-diagonal
#         N = R.shape[0]
#         mask_off = ~torch.eye(N, dtype=torch.bool)
#         off_diag_elements = R_abs[mask_off]
#         mean_coupling = torch.mean(off_diag_elements).item()
#         var_coupling = torch.var(off_diag_elements).item()
        
#         return mean_coupling, var_coupling, R, K, L_B
#     except Exception as e:
#         return None, None, None, None, None
def compute_scale_invariant_stats(target_A, basis_B):
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    if basis_B.ndim == 1: basis_B = torch.diag(basis_B)
    
    A = target_A.double().cpu()
    B = basis_B.double().cpu()
    
    try:
        L_B, V_B = torch.linalg.eigh(B)
        
        idx_flip = torch.arange(L_B.shape[0] - 1, -1, -1)
        V_B = V_B[:, idx_flip]
        L_B = L_B[idx_flip]
        
        # K: Raw projection (V_B^T @ A @ V_B)
        K = V_B.T @ A @ V_B
        
        
        diag_val = torch.abs(torch.diagonal(K))
        
        norm_factor = torch.sqrt(torch.outer(diag_val, diag_val))
        
        eps = 1e-38
        
        safe_mask = norm_factor > eps
        
        R = torch.zeros_like(K)
        R[safe_mask] = K[safe_mask] / norm_factor[safe_mask]
        
        
        R_abs = torch.abs(R) 

        N = R.shape[0]
        mask_off = ~torch.eye(N, dtype=torch.bool)
        off_diag_elements = R_abs[mask_off]
        
        mean_coupling = torch.mean(off_diag_elements).item()
        var_coupling = torch.var(off_diag_elements).item()
        
        return mean_coupling, var_coupling, R.float(), K.float(), L_B.float()
        
    except Exception as e:
        print(f"Error in stats calculation: {e}")
        return None, None, None, None, None




def compute_rmt_baseline_alignment(target_A):
    if target_A.ndim == 1: target_A = torch.diag(target_A)
    A = target_A.float().cpu()
    N = A.shape[0]
    H = torch.randn(N, N)
    Q, _ = torch.linalg.qr(H)
    A_rand = Q.T @ A @ Q
    total_energy = torch.norm(A_rand, p='fro')
    if total_energy == 0: return 0.0
    off_diag = A_rand - torch.diag_embed(torch.diag(A_rand))
    return (torch.norm(off_diag, p='fro') / total_energy).item()

# --------------------------
# --------------------------
def analyze_epochs_advanced(epoch_list, base_path, slice_range, var_pairs):
    results = {
        "slope": {k: [] for k in var_pairs},
        "spearman": {k: [] for k in var_pairs},
        "commutativity": {k: [] for k in var_pairs},
        "comm_random": {k: [] for k in var_pairs},
        "comm_random_top": {k: [] for k in var_pairs},
        "comm_random_bulk": {k: [] for k in var_pairs},
        "alignment": {k: [] for k in var_pairs}, 
        "alignment_top": {k: [] for k in var_pairs},
        "alignment_bulk": {k: [] for k in var_pairs},
        "rmt_alignment": {k: [] for k in var_pairs},
        "rmt_alignment_top": {k: [] for k in var_pairs},
        "rmt_alignment_bulk": {k: [] for k in var_pairs},
        "diag_recovery": {k: [] for k in var_pairs},
        "diag_recovery_top": {k: [] for k in var_pairs},
        "diag_recovery_bulk": {k: [] for k in var_pairs},
        "diag_recovery_random": {k: [] for k in var_pairs},
        "diag_recovery_random_top": {k: [] for k in var_pairs},
        "diag_recovery_random_bulk": {k: [] for k in var_pairs},
        "comm_top": {k: [] for k in var_pairs},
        "comm_bulk": {k: [] for k in var_pairs},
        "mean_coupling": {k: [] for k in var_pairs},
        "var_coupling": {k: [] for k in var_pairs}
    }

    last_epoch_data = {}

    for epoch in epoch_list:
        file_path = f"{base_path}{epoch}.pt"
        if not os.path.exists(file_path): continue
        try:
            loaded_data = torch.load(file_path, map_location='cpu')
        except: continue

        try:
            L_H, V_H = torch.linalg.eigh(torch.tensor(loaded_data["Hessian"]))
        except:
            V_H = None

        for name, (key_a, key_b) in var_pairs.items():
            if key_a not in loaded_data or key_b not in loaded_data: continue
            
            A_raw = torch.tensor(loaded_data[key_a]).float()
            B_raw = torch.tensor(loaded_data[key_b]).float()
            
            A, B = A_raw, B_raw
            if V_H is not None:
                 if "covar" in key_a.lower(): A = V_H.transpose(-2, -1) @ A @ V_H
                 if "covar" in key_b.lower(): B = V_H.transpose(-2, -1) @ B @ V_H
                 if "hessian" in key_a.lower(): A = V_H.transpose(-2, -1) @ A @ V_H
                 if "hessian" in key_b.lower(): B = V_H.transpose(-2, -1) @ B @ V_H
            
            A_dense, B_dense = extract_diagonals(A, B, slice_range)
            _, _, slope, r2 = compute_loglog_fit(A_dense, B_dense)
            spearman = compute_spearman_rank(A_dense, B_dense)
            
            if slope is not None: 
                results["slope"][name].append((epoch, slope))
                results["spearman"][name].append((epoch, spearman))

            comm_val = compute_commutativity(A, B)
            if comm_val is not None: 
                results["commutativity"][name].append((epoch, comm_val))
                comm_rand, top_rand, bulk_rand = compute_commutativity_random_baseline(A, B, k=100, m=1000)
                results["comm_random"][name].append((epoch, comm_rand))
                if top_rand is not None: results["comm_random_top"][name].append((epoch, top_rand))
                if bulk_rand is not None: results["comm_random_bulk"][name].append((epoch, bulk_rand))

            align_val = compute_eigen_alignment(target_A=A, basis_B=B)
            if align_val is not None: results["alignment"][name].append((epoch, align_val))
            align_top, align_bulk = compute_split_eigen_alignment(target_A=A, basis_B=B, k=100, m=1000)
            if align_top is not None: results["alignment_top"][name].append((epoch, align_top))
            if align_bulk is not None: results["alignment_bulk"][name].append((epoch, align_bulk))
            
            rmt_val, rmt_top, rmt_bulk = compute_eigen_alignment_random_baseline(A, B, k=100, m=1000)
            if rmt_val is not None: results["rmt_alignment"][name].append((epoch, rmt_val))
            if rmt_top is not None: results["rmt_alignment_top"][name].append((epoch, rmt_top))
            if rmt_bulk is not None: results["rmt_alignment_bulk"][name].append((epoch, rmt_bulk))

            diag_rec = compute_diagonal_reconstruction_recovery(target_A=A, basis_B=B)
            if diag_rec is not None: results["diag_recovery"][name].append((epoch, diag_rec))
            diag_top, diag_bulk = compute_split_diagonal_reconstruction_recovery(target_A=A, basis_B=B, k=100, m=1000)
            if diag_top is not None: results["diag_recovery_top"][name].append((epoch, diag_top))
            if diag_bulk is not None: results["diag_recovery_bulk"][name].append((epoch, diag_bulk))

            diag_rand, diag_rand_top, diag_rand_bulk = compute_reconstruction_random_baseline(A, B, k=100, m=1000)
            if diag_rand is not None: results["diag_recovery_random"][name].append((epoch, diag_rand))
            if diag_rand_top is not None: results["diag_recovery_random_top"][name].append((epoch, diag_rand_top))
            if diag_rand_bulk is not None: results["diag_recovery_random_bulk"][name].append((epoch, diag_rand_bulk))
            
            err_top, err_bulk = compute_split_commutativity(target_A=A, basis_B=B, k=100, m=1000)
            if err_top is not None:
                results["comm_top"][name].append((epoch, err_top))
                results["comm_bulk"][name].append((epoch, err_bulk))

            mean_val, var_val, R_signed, K_matrix, L_B = compute_scale_invariant_stats(target_A=A, basis_B=B)
            
            if mean_val is not None:
                results["mean_coupling"][name].append((epoch, mean_val))
                results["var_coupling"][name].append((epoch, var_val))
                
                if epoch == epoch_list[-1]:
                    if name not in last_epoch_data: last_epoch_data[name] = {}
                    last_epoch_data[name]['K_matrix'] = K_matrix
                    last_epoch_data[name]['R_matrix_signed'] = R_signed
                    last_epoch_data[name]['mean_off_diag_real'] = mean_val
                    last_epoch_data[name]['var_off_diag_real'] = var_val
                    last_epoch_data[name]['A_raw'] = A_raw
                    last_epoch_data[name]['B_raw'] = B_raw

        del loaded_data

    return results, last_epoch_data

# --------------------------
# --------------------------
def plot_all_metrics(results, config):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), constrained_layout=True)
    axes = axes.flatten()
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.10, hspace=0.12)
    
    pair_names = list(results["slope"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(pair_names)))
    color_map = {name: colors[i] for i, name in enumerate(pair_names)}

    def display_name(name):
        if "C_" in name:
            return "$\\mathbf{C}_{AWD,raw}$"
        if "Covar_" in name:
            return "$\\mathbf{Covar}$"
        if "C1_vs" in name:
            return "$\\mathbf{C}^{hh}$"
        if "C1_dia_vs" in name:
            return "$\\mathbf{C}^{hh,SD}$"
        if "C1_dia_w_dia_vs" in name:
            return "$\\mathbf{C}^{hh,SD,WD}$"
        if "H2" in name:
            return "$2\\mathbf{C}/\\sigma_w^2$"
        return name

    # --- 1. Slope ---
    ax = axes[0]
    for name, data in results["slope"].items():
        if not data: continue
        data.sort()
        epochs, vals = zip(*data)
        if "C_" in name:
            dname = "$\\mathbf{C}_{AWD,raw}$"
        elif "Covar_" in name:
            dname = "$\\mathbf{Covar}$"
        elif "C1_vs" in name:
            dname = "$\\mathbf{C}^{hh}$"
        elif "C1_dia_vs" in name:
            dname = "$\\mathbf{C}^{hh,SD}$"
        elif "C1_dia_w_dia_vs" in name:
            dname = "$\\mathbf{C}^{hh,SD,WD}$"
        elif "H2" in name:
            dname = "$2\\mathbf{C}/\\sigma_w^2$"
        ax.plot(epochs, vals, marker='o', label=dname, color=color_map[name])
    ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label=f"Lower bound = {1}")
    ax.axhline(y=2, color='g', linestyle='--', linewidth=1.5, label=f"Upper bound = {2}")
    ax.set_title("Power vs Epochs")
    ax.set_ylabel("$\gamma$")
    ax.set_ylim(0.7, 2.3)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small')

    # ax = axes[1]
    # for name in pair_names:
    #     d_full = sorted(results["commutativity"].get(name, []), key=lambda x: x[0])
    #     d_rand = sorted(results["comm_random"].get(name, []), key=lambda x: x[0]) 
    #     d_top = sorted(results["comm_top"].get(name, []), key=lambda x: x[0])
    #     d_bulk = sorted(results["comm_bulk"].get(name, []), key=lambda x: x[0])
    #     c = color_map[name]
    #     if d_full: ax.plot(*zip(*d_full), color=c, linestyle='-', linewidth=2, label=name)
    #     if d_top: ax.plot(*zip(*d_top), color=c, linestyle='--', linewidth=1.5, alpha=0.8)
    #     if d_bulk: ax.plot(*zip(*d_bulk), color=c, linestyle=':', linewidth=2, alpha=0.9)
    # if d_rand: ax.plot(*zip(*d_rand), color=c, linestyle='-.', linewidth=1.5, alpha=0.5)
    # ax.set_title("Commutativity Error\n(Solid=Full, Dashed=Top, Dotted=Bulk, DashDot=Random)")
    # ax.set_yscale('linear')
    # ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(fontsize='small', loc='upper right') 


# --- 2. Commutativity Ratio (Error / Random Baseline) ---
    ax = axes[1]
    
    # Random baselines are plotted directly as black curves with matching line styles.
    
    for name in pair_names:
        d_full = sorted(results["commutativity"].get(name, []), key=lambda x: x[0])
        d_top = sorted(results["comm_top"].get(name, []), key=lambda x: x[0])
        d_bulk = sorted(results["comm_bulk"].get(name, []), key=lambda x: x[0])
        d_rand = sorted(results["comm_random"].get(name, []), key=lambda x: x[0])
        d_rand_top = sorted(results["comm_random_top"].get(name, []), key=lambda x: x[0])
        d_rand_bulk = sorted(results["comm_random_bulk"].get(name, []), key=lambda x: x[0])
        
        if not d_rand: continue
        
        rand_map = {e: v for e, v in d_rand}
        
        c = color_map[name]
        
        def get_ratio(data_list):
            return [(e, v) for e, v in data_list if e in rand_map and v > 0]
        
        r_full = get_ratio(d_full)
        r_top = get_ratio(d_top)
        r_bulk = get_ratio(d_bulk)
        
        if r_full: 
            ax.plot(*zip(*r_full), color=c, linestyle='-', linewidth=2, marker='o', label=name)
        if r_top: 
            ax.plot(*zip(*r_top), color=c, linestyle='--', linewidth=1.5, marker='^', alpha=0.8)
        if r_bulk: 
            ax.plot(*zip(*r_bulk), color=c, linestyle=':', linewidth=2, marker='s', alpha=0.9)
        if d_rand:
            ax.plot(*zip(*[(e, v) for e, v in d_rand if v > 0]), color='black', linestyle='-', linewidth=3.2, alpha=0.75)
        if d_rand_top:
            ax.plot(*zip(*[(e, v) for e, v in d_rand_top if v > 0]), color='black', linestyle='--', linewidth=3.2, alpha=0.75)
        if d_rand_bulk:
            ax.plot(*zip(*[(e, v) for e, v in d_rand_bulk if v > 0]), color='black', linestyle=':', linewidth=3.2, alpha=0.75)

    ax.set_title(
        "Commutativity Error\n"
        r"$B=V_B\Lambda_BV_B^T,\ P_S=V_SV_S^T,\ V_S\subset V_B$"
        "\n"
        r"$\epsilon_S=\Vert P_S[A,B]P_S\Vert_F/(\Vert P_SAP_S\Vert_F\Vert P_SBP_S\Vert_F)$"
    )
    ax.set_ylabel("Error")
    
    # ax.set_ylim(bottom=0.001, top=1.2) 
    
    ax.set_yscale("linear")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    custom_lines = [
        Line2D([0], [0], color='0.35', lw=2, linestyle='-', marker='o'),
        Line2D([0], [0], color='0.35', lw=1.5, linestyle='--', marker='^'),
        Line2D([0], [0], color='0.35', lw=2, linestyle=':', marker='s'),
        Line2D([0], [0], color='black', lw=3.2, linestyle='-', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle='--', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle=':', alpha=0.75)
    ]
    style_legend = ax.legend(
        custom_lines,
        ['data full', 'data top-100', 'data bulk 101-1000',
         'spectrum-rand full', 'spectrum-rand top-100', 'spectrum-rand bulk-900'],
        title='Line style / baseline',
        loc='upper right', bbox_to_anchor=(1.0, 1.0),
        fontsize=11, title_fontsize=12,
        frameon=True, framealpha=0.9, borderaxespad=0.35
    )
    ax.add_artist(style_legend)
    color_lines = [
        Line2D([0], [0], color=color_map[name], lw=2, label=display_name(name))
        for name in pair_names
    ]
    ax.legend(
        handles=color_lines, title='Color = matrix pair',
        loc='upper right', bbox_to_anchor=(0.58, 1.0),
        fontsize=11, title_fontsize=12,
        frameon=True, framealpha=0.9, borderaxespad=0.35
    )






#############################################################################################################
# # --- 2. Commutativity Ratio (Error / Random Baseline) ---
#     ax = axes[1]
    
#     ax.axhline(y=1.0, color='gray', linestyle='-.', linewidth=1, alpha=0.5, label='Random Baseline (y=1)')
    
#     for name in pair_names:
#         d_full = sorted(results["commutativity"].get(name, []), key=lambda x: x[0])
#         d_rand = sorted(results["comm_random"].get(name, []), key=lambda x: x[0])
        
#         if not d_rand: continue
        
#         rand_map = {e: v for e, v in d_rand}
        
#         c = color_map[name]
        
#         def get_ratio(data_list):
#             return [(e, v / rand_map[e]) for e, v in data_list if e in rand_map]
        
#         r_full = get_ratio(d_full)
        
#         if r_full: 
#             ax.plot(*zip(*r_full), color=c, linestyle='-', linewidth=2, marker='o', label=name) 

#     ax.set_title("Normalized Commutativity Error\n(Ratio = Error / Random_Baseline)")
#     ax.set_ylabel("Ratio (Log Scale)")
    
    
#     ax.grid(True, linestyle='--', alpha=0.6, which='both')
    
#     ax.legend(loc='lower left', fontsize='small')
# 
##############################################################################################################






    ax = axes[2]
    for name, data in results["spearman"].items():
        if not data: continue
        data.sort()
        epochs, vals = zip(*data)
        ax.plot(epochs, vals, marker='s', markersize=4, label=name, color=color_map[name])
    ax.set_title("Spearman Rank Correlation")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Figure 4: full/top/bulk offdiag/diag RMS with spectrum-preserving random baselines.

    ax = axes[3]
    for name in pair_names:
        d_full = sorted(results["alignment"].get(name, []), key=lambda x: x[0])
        d_top = sorted(results["alignment_top"].get(name, []), key=lambda x: x[0])
        d_bulk = sorted(results["alignment_bulk"].get(name, []), key=lambda x: x[0])
        d_rand = sorted(results["rmt_alignment"].get(name, []), key=lambda x: x[0])
        d_rand_top = sorted(results["rmt_alignment_top"].get(name, []), key=lambda x: x[0])
        d_rand_bulk = sorted(results["rmt_alignment_bulk"].get(name, []), key=lambda x: x[0])
        c = color_map[name]

        if d_full:
            ax.plot(*zip(*d_full), color=c, linestyle="-", linewidth=2, marker="o", label=name)
        if d_top:
            ax.plot(*zip(*d_top), color=c, linestyle="--", linewidth=1.5, marker="^", alpha=0.8)
        if d_bulk:
            ax.plot(*zip(*d_bulk), color=c, linestyle=":", linewidth=2, marker="s", alpha=0.9)
        if d_rand:
            ax.plot(*zip(*d_rand), color="black", linestyle="-", linewidth=3.2, alpha=0.75)
        if d_rand_top:
            ax.plot(*zip(*d_rand_top), color="black", linestyle="--", linewidth=3.2, alpha=0.75)
        if d_rand_bulk:
            ax.plot(*zip(*d_rand_bulk), color="black", linestyle=":", linewidth=3.2, alpha=0.75)

    ax.set_title(
        "Offdiag/Diag RMS Ratio\n"
        r"$B=V_B\Lambda_BV_B^T,\ K_S=V_S^TAV_S$"
        "\n"
        r"$\rho_S=\mathrm{RMS}_{i\ne j}((K_S)_{ij})/\mathrm{RMS}_i((K_S)_{ii})$"
    )
    ax.set_ylabel("RMS ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    custom_lines = [
        Line2D([0], [0], color='0.35', lw=2, linestyle='-', marker='o'),
        Line2D([0], [0], color='0.35', lw=1.5, linestyle='--', marker='^'),
        Line2D([0], [0], color='0.35', lw=2, linestyle=':', marker='s'),
        Line2D([0], [0], color='black', lw=3.2, linestyle='-', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle='--', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle=':', alpha=0.75)
    ]
    style_legend = ax.legend(
        custom_lines,
        ['data full', 'data top-100', 'data bulk 101-1000',
         'spectrum-rand full', 'spectrum-rand top-100', 'spectrum-rand bulk-900'],
        title='Line style / baseline',
        loc='upper right', bbox_to_anchor=(1.0, 1.0),
        fontsize=11, title_fontsize=12,
        frameon=True, framealpha=0.9, borderaxespad=0.35
    )
    ax.add_artist(style_legend)
    color_lines = [
        Line2D([0], [0], color=color_map[name], lw=2, label=display_name(name))
        for name in pair_names
    ]
    ax.legend(
        handles=color_lines, title='Color = matrix pair',
        loc='upper right', bbox_to_anchor=(0.58, 1.0),
        fontsize=11, title_fontsize=12,
        frameon=True, framealpha=0.9, borderaxespad=0.35
    )

    for ax in axes: ax.set_xlabel("Epoch")


    save_dir = f"ICML_Figures/{config['model']}_{config['dataset']}_{config['lss_fn']}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"All_Metrics.pdf"   
    save_path = os.path.join(save_dir, filename)
    plt.savefig(
            save_path, 
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.05,
            dpi=100
        )
    plt.show()




# from somewhere import compute_scale_invariant_stats 

def plot_diagonal_reconstruction_recovery(results, config):
    if "diag_recovery" not in results:
        return

    pair_names = list(results["diag_recovery"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(pair_names)))
    color_map = {name: colors[i] for i, name in enumerate(pair_names)}

    def display_name(name):
        if "C_" in name:
            return "$\\mathbf{C}_{AWD,raw}$"
        if "Covar_" in name:
            return "$\\mathbf{Covar}$"
        if "C1_vs" in name:
            return "$\\mathbf{C}^{hh}$"
        if "C1_dia_vs" in name:
            return "$\\mathbf{C}^{hh,SD}$"
        if "C1_dia_w_dia_vs" in name:
            return "$\\mathbf{C}^{hh,SD,WD}$"
        if "H2" in name:
            return "$2\\mathbf{C}/\\sigma_w^2$"
        return name

    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)

    for name in pair_names:
        d_full = sorted(results["diag_recovery"].get(name, []), key=lambda x: x[0])
        d_top = sorted(results["diag_recovery_top"].get(name, []), key=lambda x: x[0])
        d_bulk = sorted(results["diag_recovery_bulk"].get(name, []), key=lambda x: x[0])
        d_rand = sorted(results["diag_recovery_random"].get(name, []), key=lambda x: x[0])
        d_rand_top = sorted(results["diag_recovery_random_top"].get(name, []), key=lambda x: x[0])
        d_rand_bulk = sorted(results["diag_recovery_random_bulk"].get(name, []), key=lambda x: x[0])
        c = color_map[name]

        if d_full:
            ax.plot(*zip(*d_full), color=c, linestyle="-", linewidth=2, marker="o", label=name)
        if d_top:
            ax.plot(*zip(*d_top), color=c, linestyle="--", linewidth=1.5, marker="^", alpha=0.8)
        if d_bulk:
            ax.plot(*zip(*d_bulk), color=c, linestyle=":", linewidth=2, marker="s", alpha=0.9)
        if d_rand:
            ax.plot(*zip(*d_rand), color="black", linestyle="-", linewidth=3.2, alpha=0.75)
        if d_rand_top:
            ax.plot(*zip(*d_rand_top), color="black", linestyle="--", linewidth=3.2, alpha=0.75)
        if d_rand_bulk:
            ax.plot(*zip(*d_rand_bulk), color="black", linestyle=":", linewidth=3.2, alpha=0.75)

    ax.set_title(
        "Diagonal Reconstruction Recovery\n"
        r"$B=V_B\Lambda_BV_B^T,\ K_S=V_S^TAV_S$"
        "\n"
        r"$R_S^2=\Vert\mathrm{diag}(K_S)\Vert_F^2/\Vert K_S\Vert_F^2$"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recovered energy fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    custom_lines = [
        Line2D([0], [0], color='0.35', lw=2, linestyle='-', marker='o'),
        Line2D([0], [0], color='0.35', lw=1.5, linestyle='--', marker='^'),
        Line2D([0], [0], color='0.35', lw=2, linestyle=':', marker='s'),
        Line2D([0], [0], color='black', lw=3.2, linestyle='-', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle='--', alpha=0.75),
        Line2D([0], [0], color='black', lw=3.2, linestyle=':', alpha=0.75)
    ]
    style_legend = ax.legend(
        custom_lines,
        ['data full', 'data top-100', 'data bulk 101-1000',
         'spectrum-rand full', 'spectrum-rand top-100', 'spectrum-rand bulk-900'],
        title='Line style / baseline',
        loc='upper right', fontsize=10, title_fontsize=11,
        frameon=True, framealpha=0.9
    )
    ax.add_artist(style_legend)
    color_lines = [
        Line2D([0], [0], color=color_map[name], lw=2, label=display_name(name))
        for name in pair_names
    ]
    ax.legend(
        handles=color_lines, title='Color = matrix pair',
        loc='lower left', fontsize=9, title_fontsize=10,
        frameon=True, framealpha=0.9
    )

    save_dir = f"ICML_Figures/{config['model']}_{config['dataset']}_{config['lss_fn']}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Diagonal_Reconstruction_Recovery.pdf")
    plt.savefig(
        save_path,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.05,
        dpi=100
    )
    plt.show()


def plot_snapshot_deep_dive_combined(last_epoch_data, config):
    cmap_choice = 'RdBu_r'
    
    names = list(last_epoch_data.keys())
    n_rows = len(names)
    n_cols = 3
    
    if n_rows == 0:
        print("No data to plot.")
        return

    figsize = (18, 5.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])

    print(f"Generating combined plot for {n_rows} items...")

    for idx, (name, data) in enumerate(last_epoch_data.items()):
        
        K_real = data['K_matrix']
        R_real = data['R_matrix_signed']
        mean_real = data['mean_off_diag_real']
        var_real = data['var_off_diag_real']
        A_raw = data['A_raw']
        B_raw = data['B_raw']

        try:
            L_A, _ = torch.linalg.eigh(A_raw)
            N = A_raw.shape[0]
            H = torch.randn(N, N)
            Q, _ = torch.linalg.qr(H)
            A_rand = Q @ torch.diag(L_A) @ Q.T
            mean_rand, var_rand, R_rand, _, _ = compute_scale_invariant_stats(A_rand, B_raw)
        except:
            R_rand = None
            mean_rand, var_rand = 0, 0

        disp_dim = min(2560, R_real.shape[0])

        # =========================================================
        # Plot 1: Raw Matrix K (SymLogNorm) [Column 0]
        # =========================================================
        ax = axes[idx, 0]
        ax.grid(False)
        k_data = K_real[:300, :300].numpy()
        max_val_k = np.max(np.abs(k_data))
        
        linthresh = max_val_k * 1e-3 if max_val_k > 0 else 1e-5
        norm = SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=-max_val_k*5e-2, vmax=max_val_k*5e-2, base=10)
        
        im0 = ax.imshow(k_data, cmap=cmap_choice, norm=norm, interpolation='nearest')
        
        cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        cbar0.set_label('Amplitude (SymLog)', weight='bold')

        # LaTeX Title Logic
        if "C_" in name: dname = "$\\mathbf{C}_{AWD,raw}$"
        elif "Covar_" in name: dname = "$\\mathbf{Covar}$"
        elif "C1_vs" in name: dname = "$\\mathbf{C}^{hh}$"
        elif "C1_dia_vs" in name: dname = "$\\mathbf{C}^{hh,SD}$"
        elif "C1_dia_w_dia_vs" in name: dname = "$\\mathbf{C}^{hh,SD,WD}$"
        elif "H2" in name: dname = "$2\\mathbf{C}/\\sigma_w^2$"
        else: dname = name.replace("_", " ") # Fallback

        # ax.text(-0.25, 0.5, f"{name}", transform=ax.transAxes, 
        #         rotation=90, va='center', ha='right', fontsize=14, weight='bold')
        
        ax.set_title(dname, pad=10)
        ax.set_ylabel("Basis Index")
        
        if idx == n_rows - 1:
            ax.set_xlabel("Basis Index")
        else:
            ax.set_xlabel("")

        # =========================================================
        # Plot 2: Real Correlation R [Column 1]
        # =========================================================
        ax = axes[idx, 1]
        ax.grid(False)
        r_data = R_real[:disp_dim, :disp_dim].numpy()
        im1 = ax.imshow(r_data, cmap=cmap_choice, vmin=-1, vmax=1, interpolation='nearest')
        
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Correlation', weight='bold')
        
        ax.set_title(rf"Normalized $R$ (Real)" + f"\n$\mu={mean_real:.4f} \mid \sigma^2={var_real:.2e}$", pad=10)
        ax.set_yticks([])
        
        if idx == n_rows - 1:
            ax.set_xlabel("Basis Index")

        # =========================================================
        # Plot 3: Random Baseline R [Column 2]
        # =========================================================
        ax = axes[idx, 2]
        ax.grid(False)
        if R_rand is not None:
            r_rand_data = R_rand[:disp_dim, :disp_dim].numpy()
            im2 = ax.imshow(r_rand_data, cmap=cmap_choice, vmin=-1, vmax=1, interpolation='nearest') 
            
            cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            cbar2.set_label('Correlation', weight='bold')
            
            ax.set_title(rf"Normalized $R$ (Random)" + f"\n$\mu={mean_rand:.4f} \mid \sigma^2={var_rand:.2e}$", pad=10)
            ax.set_yticks([])
            
            if idx == n_rows - 1:
                ax.set_xlabel("Basis Index")
        else:
            ax.axis('off')

    save_dir = f"ICML_Figures/{config['model']}_{config['dataset']}_{config['lss_fn']}"
    os.makedirs(save_dir, exist_ok=True)
    
    combined_name = "Combined_Deep_Dive_Snapshot.pdf"
    save_path = os.path.join(save_dir, combined_name)
    
    plt.savefig(
        save_path, 
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=100
    )
    
    print(f"Saved combined vector figure to: {save_path}")
    
    plt.show()
    plt.close()
def plot_snapshot_deep_dive_v2(last_epoch_data, config):
    cmap_choice = 'RdBu_r' 

    for name, data in last_epoch_data.items():
        print(f"\n--- Deep Dive for {name} ---")
        
        K_real = data['K_matrix']
        R_real = data['R_matrix_signed']
        mean_real = data['mean_off_diag_real']
        var_real = data['var_off_diag_real']
        A_raw = data['A_raw']
        B_raw = data['B_raw']

        try:
            L_A, _ = torch.linalg.eigh(A_raw)
            N = A_raw.shape[0]
            H = torch.randn(N, N)
            Q, _ = torch.linalg.qr(H)
            A_rand = Q @ torch.diag(L_A) @ Q.T
            mean_rand, var_rand, R_rand, _, _ = compute_scale_invariant_stats(A_rand, B_raw)
        except:
            R_rand = None
            mean_rand, var_rand = 0, 0

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        
        disp_dim = min(2560, R_real.shape[0])
        
        # ------------------------------------------------
        # Plot 1: Raw Matrix K (SymLogNorm)
        # ------------------------------------------------
        ax = axes[0]
        ax.grid(False)
        k_data = K_real[:300, :300].numpy()
        max_val_k = np.max(np.abs(k_data))
        
        linthresh = max_val_k * 1e-3 if max_val_k > 0 else 1e-5
        norm = SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=-max_val_k*5e-2, vmax=max_val_k*5e-2, base=10)
        
        im0 = ax.imshow(k_data, cmap=cmap_choice, norm=norm, interpolation='nearest')
        
        cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        cbar0.set_label('Amplitude (SymLog)', weight='bold')
        
        if "C_" in name:
            dname = "$\\mathbf{C}_{AWD,raw}$"
        elif "Covar_" in name:
            dname = "$\\mathbf{Covar}$"
        elif "C1_vs" in name:
            dname = "$\\mathbf{C}^{hh}$"
        elif "C1_dia_vs" in name:
            dname = "$\\mathbf{C}^{hh,SD}$"
        elif "C1_dia_w_dia_vs" in name:
            dname = "$\\mathbf{C}^{hh,SD,WD}$"
        elif "H2" in name:
            dname = "$2\\mathbf{C}/\\sigma_w^2$"
        ax.set_title(dname, pad=10)
        # ax.set_title(rf"Raw Covariance " + "\n(Log Scale)", pad=10)

        ax.set_xlabel("Basis Index")
        ax.set_ylabel("Basis Index ")

        # ------------------------------------------------
        # Plot 2: Real Correlation R
        # ------------------------------------------------
        ax = axes[1]
        ax.grid(False)
        r_data = R_real[:disp_dim, :disp_dim].numpy()
        im1 = ax.imshow(r_data, cmap=cmap_choice, vmin=-1, vmax=1, interpolation='nearest')
        
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Correlation', weight='bold')
        
        ax.set_title(rf"Normalized $R$ (Real)" + f"\n$\mu={mean_real:.4f} \mid \sigma^2={var_real:.2e}$", pad=10)
        ax.set_xlabel("Basis Index")
        # ax.set_ylabel("Basis Eigenmodes") 
        ax.set_yticks([])

        # ------------------------------------------------
        # Plot 3: Random Baseline R
        # ------------------------------------------------
        ax = axes[2]
        ax.grid(False)
        if R_rand is not None:
            r_rand_data = R_rand[:disp_dim, :disp_dim].numpy()
            im2 = ax.imshow(r_rand_data, cmap=cmap_choice, vmin=-1, vmax=1, interpolation='nearest') 
            
            cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            cbar2.set_label('Correlation', weight='bold')
            
            ax.set_title(rf"Normalized $R$ (Random)" + f"\n$\mu={mean_rand:.4f} \mid \sigma^2={var_rand:.2e}$", pad=10)
            ax.set_xlabel("Basis Index")
            ax.set_yticks([])
        else:
            ax.axis('off')

        # plt.suptitle(f"Deep Dive Snapshot: {name}", fontsize=16)

        save_dir = f"ICML_Figures/{config['model']}_{config['dataset']}_{config['lss_fn']}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{name}.pdf"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(
            save_path, 
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.05,
            dpi=100
        )
        
        print(f"Saved vector figure to: {save_path}")
        
        plt.show()
        plt.close()


# ============================
# ============================
if __name__ == "__main__":
    epoch_list =  [1, 10, 30, 50, 70, 80, 90, 100]#, 150, 200] 
    slice_range = torch.arange(0, 1500)
    train_size = 2000
    sample_number = 20

    net_size = 50
    n_class = 10
    config = {}
    config['lss_fn'] = 'mse'
    config['dataset'] = 'mnist' 
    config['model'] = 'FC' 
    config['net_size'] = net_size
    config['sample_holder'] = [i for i in range(n_class)]
    config['B'] = 50
    config['alpha'] = 0.1
    save_dir = f"./AWCH_data/TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"
    # save_dir = f"./AWCH_data/NS{net_size}_TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"

    file_name1 = "C_epoch_" 
    base_path1 = os.path.join(save_dir, file_name1)
    
    var_pairs = {


        "C_vs_H1": ("C", "H_1_d"),
        "C1_vs_H1": ("C1", "H_1_d"),
        "C1_dia_vs_H1": ("C1_dia", "H_1_d"),
        "C1_dia_w_dia_vs_H1": ("C1_dia_w_dia", "H_1_d"),
        "Covar_vs_Hessian": ("Covar", "Hessian"),
        "H2_vs_H1": ("H_2_d", "H_1_d"),

    }

    print(" ( Spearman, Combined Commutativity)...")
    full_results, last_epoch_data = analyze_epochs_advanced(epoch_list, base_path1, slice_range, var_pairs)
    plot_all_metrics(full_results, config)
    plot_diagonal_reconstruction_recovery(full_results, config)
    if last_epoch_data:
        plot_snapshot_deep_dive_combined(last_epoch_data, config)
    pass

    file_path1 = f"{base_path1}{100}.pt"
    loaded_data1 = torch.load(file_path1)
    
    if 'train_loss_holder' in loaded_data1:
        train_loss_holder = loaded_data1['train_loss_holder']
        test_loss_holder = loaded_data1['test_loss_holder']
        train_accuracy_holder = loaded_data1['train_accuracy_holder']
        test_accuracy_holder = loaded_data1['test_accuracy_holder']
        plt.plot(np.log10(train_loss_holder))
        plt.plot(np.log10(test_loss_holder))
        plt.title('loss')
        plt.legend(['train','test'])
        plt.show()
        plt.plot(train_accuracy_holder)
        plt.plot(test_accuracy_holder)
        plt.title('accuracy')
        plt.legend(['train','test'])
        plt.show()

# # ============================
# # Multilayer Analysis
# # ============================
# train_size = 2000
# sample_number = 20
# n_class = 10

# hidden_sizes = [50, 50, 50, 50]
# layer_indices = [1, 2, 3, 4]
# config_multi = {}
# config_multi['lss_fn'] = 'cse'
# config_multi['dataset'] = 'mnist'
# config_multi['model'] = 'FC_multilayer'
# config_multi['hidden_sizes'] = hidden_sizes
# config_multi['layer_index'] = layer_indices
# config_multi['sample_holder'] = [i for i in range(n_class)]
# config_multi['B'] = 50
# config_multi['alpha'] = 0.1

# save_dir_multi = (
#     f"./AWCH_data/HS{hidden_sizes}_layer{layer_indices}"
#     f"_TrainSize{train_size}_SampleN{sample_number}"
#     f"_ClassN{len(config_multi['sample_holder'])}"
#     f"_B{config_multi['B']}lr{config_multi['alpha']}"
#     f"_lossfn_{config_multi['lss_fn']}"
#     f"_model_{config_multi['model']}"
#     f"_dataset_{config_multi['dataset']}"
# )
# base_path_multi = os.path.join(save_dir_multi, "H1_H2_epoch_")

# var_pairs_multi = {
#     "Covar_vs_Hessian": ("Covar", "Hessian"),
#     "H2_vs_H1": ("H_2_d", "H_1_d"),
# }


# print("\n\n========== Multilayer Analysis ==========")
# print(f"save_dir: {save_dir_multi}")
# full_results_multi, last_epoch_data_multi = analyze_epochs_advanced(
#     epoch_list_multi, base_path_multi, None, var_pairs_multi
# )

# # --- Print commutativity results ---
# for name in var_pairs_multi:
#     print(f'\n=== {name} ===')
#     for metric in ['commutativity', 'comm_random', 'comm_top', 'comm_bulk']:
#         vals = full_results_multi[metric].get(name, [])
#         for epoch, v in vals:
#             print(f'  {metric:20s}  epoch={epoch:4d}  value={v:.6e}')
#     d_full = dict(full_results_multi['commutativity'].get(name, []))
#     d_rand = dict(full_results_multi['comm_random'].get(name, []))
#     for e in sorted(d_full):
#         if e in d_rand and d_rand[e] != 0:
#             print(f'  ratio(full/rand)      epoch={e:4d}  value={d_full[e]/d_rand[e]:.6e}')

# plot_all_metrics(full_results_multi, config_multi)
# if last_epoch_data_multi:
#     plot_snapshot_deep_dive_combined(last_epoch_data_multi, config_multi)

# file_path_multi = f"{base_path_multi}{epoch_list_multi[-1]}.pt"
# if os.path.exists(file_path_multi):
#     loaded_data_multi = torch.load(file_path_multi, map_location='cpu')
#     if 'train_loss_holder' in loaded_data_multi:
#         plt.plot(np.log10(loaded_data_multi['train_loss_holder']))
#         plt.plot(np.log10(loaded_data_multi['test_loss_holder']))
#         plt.title('Multilayer Loss')
#         plt.legend(['train', 'test'])
#         plt.show()
#         plt.plot(loaded_data_multi['train_accuracy_holder'])
#         plt.plot(loaded_data_multi['test_accuracy_holder'])
#         plt.title('Multilayer Accuracy')
#         plt.legend(['train', 'test'])
#         plt.show()
# # ============================

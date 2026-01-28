import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.func import hessian, grad, vmap, functional_call
import time
import gc
import torch.nn.functional as F  



def get_hidden_output(model, data_x, layer_index):
    _ = model(data_x) 

    feature = model.feature[layer_index]
    return feature

def get_target_params_and_buffers_C(model, layer_index):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    target_key = list(params.keys())[layer_index]
    target_params = {target_key: params[target_key]}
    other_params = {k: v for k, v in params.items() if k != target_key}
    return target_params, other_params, buffers, target_key





def pre_calculate_all_features_custom(model, x, layer_index, batch_size=64):
    device = x.device
    temp_features = []
    
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print("Pre-calculating features using `get_hidden_output`...")
    
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            feat = get_hidden_output(model, bx, layer_index)
            
            if isinstance(feat, list):
                feat = feat[0]
                
            temp_features.append(feat.detach().cpu())
            
    return torch.cat(temp_features, dim=0).view(x.shape[0], -1)

def cal_C_cuda(model, train_x, train_y, sample_holder, layer_index, components, 
               batch_size=50, sample_number=10, loss_fn_name='cse', iterations=None):

    device = train_x.device
    model.zero_grad(set_to_none=True)
    model.eval()
    
    if not isinstance(components, torch.Tensor):
        components = torch.tensor(components, dtype=torch.float32)
    components = components.to(device)
    
    target_weight = list(model.parameters())[layer_index]
    D_out, D_in = target_weight.shape
    D_flat = D_out * D_in
    
    assert components.shape == (D_flat, D_flat), "Components shape mismatch!"
    print(f"Processing Layer {layer_index} | D_flat: {D_flat}")

    class_indices_map = {}
    for cls in sample_holder:
        indices = torch.nonzero(train_y == cls, as_tuple=True)[0]
        class_indices_map[cls] = indices
    
    all_cached_features = pre_calculate_all_features_custom(model, train_x, layer_index)
    
    mask = torch.isin(train_y, torch.tensor(sample_holder, device=device))
    valid_indices = torch.nonzero(mask, as_tuple=True)[0]

    target_params, other_params, buffers, target_key = get_target_params_and_buffers_C(model, layer_index)
    
    with torch.no_grad():
        dummy = model(train_x[0:1].to(device))
        num_classes = dummy.shape[-1]

    def compute_loss_stateless(t_params, x, y):
        all_p = {**other_params, **t_params}
        out = functional_call(model, (all_p, buffers), (x.unsqueeze(0),))
        if loss_fn_name == 'mse':
            loss = F.mse_loss(F.softmax(out, dim=-1), y.unsqueeze(0))
        elif loss_fn_name == 'lmse':
            loss = F.mse_loss(out, y.unsqueeze(0))
        elif loss_fn_name == 'cse':
            loss = F.cross_entropy(out, y.unsqueeze(0))
        return loss

    batch_grad_fn = vmap(lambda p, x, y: grad(compute_loss_stateless)(p, x, y)[target_key].view(-1), in_dims=(None, 0, 0))
    batch_hess_fn = vmap(lambda p, x, y: hessian(compute_loss_stateless)(p, x, y)[target_key][target_key].view(D_flat, D_flat), in_dims=(None, 0, 0))

    acc_C1_dia = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C2_dia = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C3_dia = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C1 = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C2 = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C3 = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C1_dia_w_dia = torch.zeros((D_flat, D_flat), device='cpu')
    acc_C1_h = torch.zeros((D_flat, D_flat), device='cpu')
    acc_H1 = torch.zeros((D_flat, D_flat), device='cpu')
    acc_H2 = torch.zeros((D_flat, D_flat), device='cpu')
    

    dataset = TensorDataset(train_x[valid_indices], train_y[valid_indices], valid_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_processed = 0
    start_time = time.time()
    
    print("Starting Loop...")

    for batch_idx, (b_x, b_y, b_global_idx) in enumerate(loader):
        if iterations is not None and batch_idx >= iterations:
            break
            
        B = b_x.shape[0]
        b_x, b_y = b_x.to(device), b_y.to(device)
        
        anchor_indices_list = []
        anchor_labels_list = []  
        
        for cls in sample_holder:
            idx_pool = class_indices_map[cls]
            perm = torch.randperm(len(idx_pool))
            k = min(len(idx_pool), sample_number)
            chosen = idx_pool[perm[:k]]
            
            anchor_indices_list.append(chosen)
            
            labels = torch.full((len(chosen),), cls, dtype=torch.long, device=device)
            anchor_labels_list.append(labels)
            
        anchor_indices = torch.cat(anchor_indices_list)
        anchor_labels = torch.cat(anchor_labels_list) 
        
        anchor_features = all_cached_features[anchor_indices.cpu()].to(device)

        y_in = F.one_hot(b_y, num_classes).float() if "mse" in loss_fn_name.lower() else b_y
        raw_g = batch_grad_fn(target_params, b_x, y_in).detach()
        raw_h = batch_hess_fn(target_params, b_x, y_in).detach()

        b_features = all_cached_features[b_global_idx.cpu()].to(device)
        
        dists = torch.cdist(b_features, anchor_features)
        

        class_diff_mask = b_y.unsqueeze(1) != anchor_labels.unsqueeze(0)
        
        dists.masked_fill_(class_diff_mask, float('inf'))

        vals, inds = torch.topk(dists, k=2, dim=1, largest=False)
        
        is_self = vals[:, 0] < 1e-6
        closest_idx_in_anchor = torch.where(is_self, inds[:, 1], inds[:, 0])
        
        x_train = b_features  # (B, D_in)
        delta_x = anchor_features[closest_idx_in_anchor] - x_train  # (B, D_in)
        Z = torch.sum(x_train**2, dim=1, keepdim=True) 

        # w (B, D_flat)
        term_wd = F.linear(delta_x, target_weight)
        w_raw = torch.einsum('bo, bi -> boi', term_wd, x_train).reshape(B, -1) / Z
        
        # a (B, D_flat, D_flat) - Kronecker Product via Einsum
        sub_block = torch.einsum('bj, bn -> bjn', x_train, delta_x) / Z.unsqueeze(-1)
        eye = torch.eye(D_out, device=device).unsqueeze(0).expand(B, -1, -1)
        a_raw = torch.einsum('bxy, buv -> bxuyv', eye, sub_block).reshape(B, D_flat, D_flat)

        h_rot = torch.matmul(torch.matmul(components, raw_h), components.T)
        g_rot = raw_g @ components.T
        w_rot = w_raw @ components.T
        a_rot = torch.matmul(torch.matmul(components, a_raw), components.T)
        
        del raw_h, raw_g, w_raw, a_raw, sub_block
        
        gc.collect()
        torch.cuda.empty_cache()

        h, g, w, a = h_rot, g_rot, w_rot, a_rot
        del h_rot, g_rot, w_rot, a_rot
        torch.cuda.empty_cache()

        # Dia Terms
        acc_C1_dia += torch.einsum('smi,si,sj,sjn->mn', h, w, w, h).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        acc_C2_dia += torch.einsum('smi,si,sj,sjn->mn', h, w, g, a).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        acc_C3_dia += torch.einsum('si,sim,sj,sjn->mn', g, a, g, a).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        acc_C1 += torch.einsum('smi,si,kj,kjn->mn', h, w, w, h).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        acc_C2 += torch.einsum('smi,si,kj,kjn->mn', h, w, g, a).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()
        acc_C3 += torch.einsum('si,sim,kj,kjn->mn', g, a, g, a).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.preferred_linalg_library('magma')
        L, v = torch.linalg.eigh(h)
        p = torch.matmul(v.transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
        mid_diag = (L * p) ** 2
        v_scaled = v * mid_diag.unsqueeze(1)
        acc_C1_dia_w_dia += torch.sum(torch.matmul(v_scaled, v.transpose(-2, -1)), dim=0).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # acc_C1_dia_w_dia += torch.einsum('smi,si,si,sin->mn', h, w, w, h).detach().cpu()
        acc_C1_h += torch.einsum('smi,sin->mn', h, h).detach().cpu()
        
        acc_H1 += h.sum(0).detach().cpu()
        acc_H2 += torch.matmul(h, h).sum(0).detach().cpu()
        
        
        total_processed += B
        del h, g, w, a
        gc.collect()
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx+1} | Time: {time.time() - start_time:.2f}s")

    N = float(total_processed)
    
    C1_dia = acc_C1_dia.numpy() / N
    C2_dia = acc_C2_dia.numpy() / N
    C3_dia = acc_C3_dia.numpy() / N
    C1 = acc_C1.numpy() / N
    C2 = acc_C2.numpy() / N
    C3 = acc_C3.numpy() / N
    C1_dia_w_dia = acc_C1_dia_w_dia.numpy() / N
    C1_h = acc_C1_h.numpy() / N
    H_1_d = acc_H1.numpy() / N
    H_2_d = acc_H2.numpy() / N
    
    # Mean Field Global Terms: C1 = E[Hw] @ E[Hw].T



    return C1_dia/batch_size, C2_dia/batch_size, C3_dia/batch_size, C1/batch_size, C2/batch_size, C3/batch_size, C1_dia_w_dia/batch_size, C1_h/batch_size, H_1_d, H_2_d/batch_size




def get_target_params_and_buffers(model, layer_index):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    valid_keys = [n for n, p in model.named_parameters()]
    if layer_index >= len(valid_keys):
        raise ValueError(f"Layer index {layer_index} out of range.")
    target_key = valid_keys[layer_index]
    print(f"Target Parameter: {target_key} | Shape: {params[target_key].shape}")
    target_params = {target_key: params[target_key]}
    other_params = {k: v for k, v in params.items() if k != target_key}
    return target_params, other_params, buffers, target_key
 

def cal_hessian_stats_cuda(model, train_x, train_y, components, layer_index, loss_fn, batch_size=32):
    device = train_x.device
    model.eval()
    
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    components = torch.tensor(components, dtype=torch.float32, device=device)
    K = components.shape[0]
    
    H1_sum = torch.zeros((K, K), device=device)
    H2_sum = torch.zeros((K, K), device=device)
    total_samples = 0
    
    target_params, other_params, buffers, target_key_name = get_target_params_and_buffers(model, layer_index)
    
    with torch.no_grad():
        dummy_out = model(train_x[0:1].to(device))
        num_classes = dummy_out.shape[-1]
    
    def compute_loss_stateless(target_params_arg, x_arg, y_arg):
        all_params = {**other_params, **target_params_arg}
        out = functional_call(model, (all_params, buffers), (x_arg.unsqueeze(0),))
        

        if isinstance(loss_fn, str):
            if loss_fn == 'mse':
                probs = F.softmax(out, dim=-1)
                loss = F.mse_loss(probs, y_arg.unsqueeze(0))
            elif loss_fn == 'lmse':
                loss = F.mse_loss(out, y_arg.unsqueeze(0))
            elif loss_fn == 'cse':
                loss = F.cross_entropy(out, y_arg.unsqueeze(0))
        else:
            loss = loss_fn(out, y_arg.unsqueeze(0))
            
        return loss

    def get_single_hessian_flat(params, x, y):
        h_dict = hessian(compute_loss_stateless)(params, x, y)
        
        h_raw = h_dict[target_key_name][target_key_name]
        
        total_dim = int(np.sqrt(h_raw.numel()))
        return h_raw.view(total_dim, total_dim)

    batch_hessian_fn = vmap(get_single_hessian_flat, in_dims=(None, 0, 0))

    print(f"--- Processing {len(dataset)} samples ---")
    start_time = time.time()
    
    for batch_idx, (b_x, b_y) in enumerate(loader):
        b_x, b_y = b_x.to(device), b_y.to(device)
        current_bs = b_x.shape[0]
        
        if "mse" in loss_fn.lower():
            b_y_input = F.one_hot(b_y, num_classes=num_classes).float()
        else:
            b_y_input = b_y
        raw_H = batch_hessian_fn(target_params, b_x, b_y_input)
        
        #
        H_proj = torch.einsum('kd, bde, le -> bkl', components, raw_H, components)

        del raw_H


        current_H1 = H_proj.sum(dim=0)
        H1_sum += current_H1.detach() 
        
        del current_H1
        H_sq = torch.bmm(H_proj, H_proj)
        current_H2 = H_sq.sum(dim=0)
        H2_sum += current_H2.detach()



        del H_proj, current_H2, H_sq

        
        total_samples += current_bs
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)} | Time: {time.time() - start_time:.1f}s")
        
        torch.cuda.empty_cache()
        gc.collect()
            
    H_1_d = (H1_sum / total_samples).detach().cpu().numpy()
    H_2_d = (H2_sum / total_samples).detach().cpu().numpy()
    
    return H_1_d, H_2_d



########################################################



def cal_h_g_cuda(model, train_x, train_y, sample_holder, layer_index, components, sample_number, loss_fn_name, batch_size=8):

    device = train_x.device
    model.eval()

    print("Filtering data...")
    selected_x = []
    selected_y = []
    
    for class_id in sample_holder:

        indices = torch.nonzero(train_y == class_id, as_tuple=True)[0]
        if len(indices) < sample_number:
            print(f"Warning: Class {class_id} has only {len(indices)} samples, requested {sample_number}.")
            current_indices = indices
        else:
            current_indices = indices[:sample_number]
            
        selected_x.append(train_x[current_indices])
        selected_y.append(train_y[current_indices])
    
    if not selected_x:
        raise ValueError("No samples selected.")
        
    target_x = torch.cat(selected_x, dim=0) # (N_total, D_in)
    target_y = torch.cat(selected_y, dim=0) # (N_total,)
    
    N_total = target_x.shape[0]
    print(f"Total samples to process: {N_total}")


    dataset = TensorDataset(target_x, target_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    target_params, other_params, buffers, target_key_name = get_target_params_and_buffers(model, layer_index)

    with torch.no_grad():
        dummy_out = model(target_x[0:1].to(device))
        num_classes = dummy_out.shape[-1]

    def compute_loss_stateless(target_params_arg, x_arg, y_arg):
        all_params = {**other_params, **target_params_arg}
        out = functional_call(model, (all_params, buffers), (x_arg.unsqueeze(0),))
        
        if loss_fn_name == 'mse':
            # Softmax -> MSE
            probs = F.softmax(out, dim=-1)
            loss = F.mse_loss(probs, y_arg.unsqueeze(0))
        elif loss_fn_name == 'lmse':
            loss = F.mse_loss(out, y_arg.unsqueeze(0))
        elif loss_fn_name == 'cse':
            loss = F.cross_entropy(out, y_arg.unsqueeze(0))
        else:
            loss = loss_fn_name(out, y_arg.unsqueeze(0))
        return loss

    def get_single_grad_flat(params, x, y):
        g_dict = grad(compute_loss_stateless)(params, x, y)
        g_raw = g_dict[target_key_name]
        return g_raw.view(-1) # Flatten to vector

    def get_single_hessian_flat(params, x, y):
        h_dict = hessian(compute_loss_stateless)(params, x, y)
        h_raw = h_dict[target_key_name][target_key_name]
        total_dim = int(np.sqrt(h_raw.numel()))
        return h_raw.view(total_dim, total_dim) # Flatten to Matrix

    batch_grad_fn = vmap(get_single_grad_flat, in_dims=(None, 0, 0))
    batch_hessian_fn = vmap(get_single_hessian_flat, in_dims=(None, 0, 0))


    h_holder_list = []
    g_holder_list = []
    
    print(f"Start processing on {device}...")
    start_time = time.time()

    for batch_idx, (b_x, b_y) in enumerate(loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        
        # --- Pre-process Y for MSE ---
        if "mse" in loss_fn_name.lower():
            b_y_input = F.one_hot(b_y, num_classes=num_classes).float()
        else:
            b_y_input = b_y

        # raw_g shape: (Batch, D_flat)
        raw_g = batch_grad_fn(target_params, b_x, b_y_input)
        
        # raw_h shape: (Batch, D_flat, D_flat)
        raw_h = batch_hessian_fn(target_params, b_x, b_y_input)
        

        g_holder_list.append(raw_g.detach().cpu().numpy())
        h_holder_list.append(raw_h.detach().cpu().numpy())
        
        del raw_g, raw_h, b_y_input
        torch.cuda.empty_cache()
        gc.collect() 
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)} | Time: {time.time() - start_time:.1f}s")


    print("Concatenating results...")
    g_final = np.concatenate(g_holder_list, axis=0)
    
    h_final = np.concatenate(h_holder_list, axis=0)
    
    print(f"Finished. g shape: {g_final.shape}, h shape: {h_final.shape}")
    return h_final, g_final
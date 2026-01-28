import torch
import numpy as np
import torch.nn as nn
import copy
import torch.utils.data as Data
import train_model
import gc
import torch.nn.functional as F
from torch.func import functional_call, hessian
from torch.utils.data import DataLoader, TensorDataset


def transform_matrix_to_array(para_list):
    para_holder = []  # Initialize an empty list to store flattened tensors

    # Iterate through each tensor in para_list
    for para in para_list:
        # Convert the PyTorch tensor to a NumPy array, flatten it to 1D, and append to para_holder
        para_holder.append(para.detach().cpu().clone().numpy().reshape(1, -1))

    # Concatenate the flattened arrays horizontally (stack them)
    para_array = np.hstack(para_holder)

    return para_array  # Return the horizontally stacked NumPy array


def transform_array_to_matrix(model, layer_index, para_array):
    para_list = []  # Initialize an empty list to store the transformed tensors
    start_point = 0  # Initialize the starting point for slicing the flattened array

    # Iterate through each layer index in the layer_index list
    for i in layer_index:
        weight = list(model.parameters())[i]  # Retrieve the parameters (weights) of the specified layer
        num_weight = np.prod(weight.shape)  # Calculate the total number of elements in the weight tensor

        # Slice the flattened array and reshape it to match the shape of the weight tensor
        para_matrix = para_array[0][start_point:num_weight + start_point].reshape(weight.shape)

        # Convert the sliced and reshaped array into a PyTorch tensor and append it to para_list
        para_list.append(torch.tensor(para_matrix))

        # Update the start_point for the next layer by adding the number of elements processed
        start_point += num_weight

    return para_list  # Return the list of transformed tensors


def replace_weight(model,weight_list):
    paras_dict = model.state_dict()
    paras_key = list(paras_dict.keys())
    for i in range(len(paras_key)):
        key_name = paras_key[i]
        paras_dict[key_name] = weight_list[i]
    model.load_state_dict(paras_dict)
    return model

def init_weight(model,scale_ratio):
    weight_list = list(model.parameters())
    scaled_weight_list = scale_weight_list(weight_list,scale_ratio)
    model = replace_weight(model,scaled_weight_list)
    return model

def scale_weight_list(weight_list,scale_ratio):
    scaled_weight_list = []
    for i in range(len(weight_list)):
        scaled_weight_list.append(weight_list[i]*scale_ratio)
    return scaled_weight_list

# def predict_accuracy(model,data_x,data_y):


#     pred = torch.max(model(data_x),1)[1].detach().cpu().numpy()
#     accuracy = np.mean(pred == data_y.detach().cpu().numpy())


    
#     return accuracy


def predict_accuracy(model, data_x, data_y, batch_size=30):
    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data_x), batch_size):
            batch_x = data_x[i:i+batch_size].to(device)
            batch_y = data_y[i:i+batch_size].to(device)
            
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            
            correct += (preds == batch_y).sum().item()
            total += len(batch_x)

    accuracy = correct / total
    return accuracy





def cal_loss(model, data_x, data_y, loss_fn, batch_size=30):

    device = next(model.parameters()).device  
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad(): 
        for i in range(0, len(data_x), batch_size):
            batch_x = data_x[i:i+batch_size].to(device)
            batch_y = data_y[i:i+batch_size].to(device)
            
            output = model(batch_x)

            if loss_fn == 'mse':
                y_onehot = nn.functional.one_hot(batch_y, num_classes=output.shape[-1]).float()
                loss_func = nn.MSELoss()

                probs = nn.functional.softmax(output, dim=-1)
                loss = loss_func(probs,y_onehot) 


            elif loss_fn == 'cse':
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(output, batch_y)

            elif loss_fn == 'lmse':
                loss_func = nn.MSELoss()
                y_onehot = nn.functional.one_hot(batch_y, num_classes=output.shape[-1]).float()
                loss = loss_func(output, y_onehot)
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")

            total_loss += loss.item() * len(batch_x)
            total_samples += len(batch_x)

    return total_loss / total_samples



def cal_grad_for_given_coordinate(model,components,data_x,data_y,layer_index, loss_fn):
    
    copy_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copy_model.parameters(),lr=0.1)
    out_put = model(data_x)
    if loss_fn == 'mse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        probs = nn.functional.softmax(out_put, dim=-1)
        l = loss_func(probs,y) 
    elif loss_fn == 'lmse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        l = loss_func(out_put,y)
    elif loss_fn == 'cse':
        loss_func = nn.CrossEntropyLoss()
        l = loss_func(out_put,data_y)


    optimizer.zero_grad()
    l.backward()
    grads_matrix = [list(copy_model.parameters())[l].grad.detach() for l in layer_index]
    grads = transform_matrix_to_array(grads_matrix)
    new_grads = np.dot(grads,components.T)
    
    return new_grads    




def smooth_curve(sequence,window_size):
    ave_sequence = []
    for i in range(len(sequence)-window_size):
        ave_sequence.append(np.mean(sequence[i:i+window_size]))
    return ave_sequence

def regularization(model, threshold):
    device = next(model.parameters()).device

    all_params = torch.cat([p.view(-1) for p in model.parameters()]).to(device)

    norm = torch.norm(all_params, p=2)
    R = torch.relu(norm - threshold)

    return R


def get_net_paras(model):
    model_copy = copy.deepcopy(model)
    weight_list = list(model_copy.parameters())
    grad_list = []
    for i in range(len(weight_list)):
        grad_list.append(weight_list[i].grad)
    return weight_list,grad_list

def cal_L2_norm_weight(para_list):
    w = transform_matrix_to_array(para_list)
    norm = np.linalg.norm(w)
    return norm

def cal_L2_norm_model(model):
    weight_list,_ = get_net_paras(model)
    weight_norm = cal_L2_norm_weight(weight_list)
    return weight_norm

def cal_fisher_information(model,data_x,data_y,layer_index,batch_size,loss_fn):
    from scipy import linalg
    torch_dataset = Data.TensorDataset(data_x,data_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size = batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    sample_grad_holder = []
    for step, (b_x,b_y) in enumerate(train_loader):
        optimizer.zero_grad()
        out_put = model(b_x)
        if loss_fn == 'mse':
            y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
            probs = nn.functional.softmax(out_put, dim=-1)
            loss = nn.MSELoss()(probs,y)
        elif loss_fn == 'lmse':
            y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
            loss = nn.MSELoss()(out_put,y)
        elif loss_fn == 'cse':
            loss = nn.CrossEntropyLoss()(out_put,b_y)

        loss.backward()
        
        layer_weight_grad = []
        for i in layer_index:
            layer_weight_grad.append(list(model.parameters())[i].grad)
        
        
        sample_grad = transform_matrix_to_array(layer_weight_grad)
        sample_grad_holder.append(sample_grad)
    
    fisher_info_matrix = 0
    for grad in sample_grad_holder:
        fisher_info_matrix = fisher_info_matrix + np.dot(grad.T,grad)
    fisher_info_matrix = fisher_info_matrix/len(sample_grad_holder)
    
    v,w = linalg.eigh(fisher_info_matrix)
    
    return fisher_info_matrix,np.real(v),np.real(w).T



def cal_noise_covar(model,data_x,data_y,layer_index,batch_size,loss_fn):

    L_grad = cal_grad(model,data_x,data_y,layer_index,loss_fn)
    L_grad = np.squeeze(L_grad)
    F,v,w = cal_fisher_information(model,data_x,data_y,layer_index,batch_size,loss_fn)
    C = F - np.outer(L_grad.T,L_grad)
    return C

def cal_noise_covar_minibatch(model,data_x,data_y,layer_index,batch_size,loss_fn):

    L_grad = cal_grad_minibatch(model,data_x,data_y,layer_index,loss_fn)
    L_grad = np.squeeze(L_grad)
    F,v,w = cal_fisher_information(model,data_x,data_y,layer_index,batch_size,loss_fn)
    C = F - np.outer(L_grad.T,L_grad)
    return C


def cal_commutation(x,y):
    c = np.linalg.norm(np.dot(x,y)-np.dot(y,x))/(np.linalg.norm(x)*np.linalg.norm(y))
    eigenvalue,eigenvector = np.linalg.eig(y)

    x_diag = np.diagonal(np.dot(eigenvector,np.dot(x,eigenvector.T)))
    x_til = np.zeros(x.shape)
    for i in range(len(x_diag)):
        x_til[i,i] = x_diag[i]
    x_reconstructed = np.diagonal(np.dot(eigenvector.T,np.dot(x_til,eigenvector)))

    R = 1 - np.sum(np.sum(x-x_reconstructed))/np.sum(np.sum(x-np.sum(np.sum(x,0))/np.prod(x.shape)))
    return c,R



def cal_grad_minibatch(model, data_x, data_y, layer_index, loss_fn, batch_size=128):
    device = next(model.parameters()).device
    model = copy.deepcopy(model).to(device)
    

    params = [list(model.parameters())[l] for l in layer_index]
    P = sum(p.numel() for p in params)  
    
    grad_sum = torch.zeros(P, device=device)
    
    N = len(data_x)
    num_batches = (N + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        print(f"Processing batch {b+1}/{num_batches} for grads computation...")
        batch_x = data_x[b*batch_size:(b+1)*batch_size].to(device)
        batch_y = data_y[b*batch_size:(b+1)*batch_size].to(device)
        
        # forward
        out = model(batch_x)
        
        # loss
        if loss_fn == 'mse':
            y = nn.functional.one_hot(batch_y, num_classes=out.shape[-1]).float()
            probs = nn.functional.softmax(out, dim=-1)
            loss = nn.MSELoss()(probs, y)
        elif loss_fn == 'lmse':
            y = nn.functional.one_hot(batch_y, num_classes=out.shape[-1]).float()
            loss = nn.MSELoss()(out, y)
        elif loss_fn == 'cse':
            loss = nn.CrossEntropyLoss()(out, batch_y)
        

        grads_list = torch.autograd.grad(loss, params)

        grads_flat = torch.cat([g.reshape(-1) for g in grads_list])

        grad_sum += grads_flat * (len(batch_x) / N)

        del batch_x, batch_y, out, loss, grads_list, grads_flat
        torch.cuda.empty_cache()
    
    return grad_sum.detach().cpu().numpy()



def cal_grad(model,data_x,data_y,layer_index,loss_fn):

    copy_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copy_model.parameters(),lr=0.1)
    out_put = copy_model(data_x)
    if loss_fn == 'mse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        probs = nn.functional.softmax(out_put, dim=-1)
        loss = loss_func(probs,y)
    elif loss_fn == 'lmse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        loss = loss_func(out_put,y)
    elif loss_fn == 'cse':
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out_put,data_y)

    optimizer.zero_grad()
    loss.backward()
    grads_matrix = [list(copy_model.parameters())[l].grad.detach() for l in layer_index]
    grads = transform_matrix_to_array(grads_matrix)
    
    return grads

def cal_hessian(model,data_x,data_y, layer_index, loss_fn):
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    layer = layer_index[0]
    parameter = list(model.parameters())[layer]
    out_put = model(data_x)
    if loss_fn == 'mse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        probs = nn.functional.softmax(out_put, dim=-1)
        loss = loss_func(probs,y)
    elif loss_fn == 'lmse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        loss = loss_func(out_put,y)
    elif loss_fn == 'cse':
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out_put,data_y)



    grad_1 = torch.autograd.grad(loss,parameter,create_graph=True)[0]
    hessian = []
    for grad in grad_1.view(-1):
        grad_2 = torch.autograd.grad(grad,parameter,create_graph=True)[0].view(-1)
        hessian.append(grad_2.detach().cpu().numpy())

    h = np.array(hessian)

    
    eigenvalue,eigenvector = torch.linalg.eigh(torch.tensor(h,device=device))
    eigenvalue = eigenvalue.detach().cpu().numpy()
    eigenvector = eigenvector.detach().cpu().numpy()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.to('cpu')
        param.to('cpu')
        del param.grad
        del param

    loss.to('cpu')
    loss = None
    model = None
    del model
    del loss
    gc.collect(generation=2)
    torch.cuda.empty_cache()


    return hessian,np.real(eigenvalue),np.real(eigenvector).T



def get_target_params_and_buffers(model, layer_index):

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    target_key = list(params.keys())[layer_index[0]]
    
    target_params = {target_key: params[target_key]}
    other_params = {k: v for k, v in params.items() if k != target_key}
    
    return target_params, other_params, buffers, target_key

def cal_hessian_cuda(model, data_x, data_y, layer_index, loss_fn, batch_size=64):

    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    

    N = len(data_x)
    dataset = TensorDataset(data_x, data_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    target_params, other_params, buffers, target_key = get_target_params_and_buffers(model, layer_index)

    P = target_params[target_key].numel()
    D_shape = target_params[target_key].shape
    print(f"Calculating Hessian for Layer {layer_index[0]} | Params: {P} | Shape: {D_shape}")
    

    hessian_sum = torch.zeros((P, P), device=device)

    with torch.no_grad():
        dummy_out = model(data_x[0:1].to(device))
        num_classes = dummy_out.shape[-1]


    def compute_loss_stateless(t_params, x, y):
        all_params = {**other_params, **t_params}
        out = functional_call(model, (all_params, buffers), (x,))
        
        if loss_fn == "mse":

            probs = F.softmax(out, dim=-1)

            loss = F.mse_loss(probs, y)
        elif loss_fn == "lmse":

            loss = F.mse_loss(out, y)
        elif loss_fn == "cse":

            loss = F.cross_entropy(out, y)
        return loss


    def get_hessian_matrix(t_params, x, y):
        h_dict = hessian(compute_loss_stateless)(t_params, x, y)
        h_raw = h_dict[target_key][target_key]
        return h_raw.view(P, P)

    print(f"Total samples: {N}. Batch size: {batch_size}.")
    

    for b_idx, (b_x, b_y) in enumerate(loader):
        print(f"Processing batch {b_idx+1}/{len(loader)}...")
        b_x = b_x.to(device)
        b_y = b_y.to(device)


        if  "mse" in loss_fn.lower():
            y_in = F.one_hot(b_y, num_classes=num_classes).float()
        else:
            y_in = b_y 

        H_batch = get_hessian_matrix(target_params, b_x, y_in)
        
        weight = len(b_x) / N
        hessian_sum += weight * H_batch.detach() 


        del H_batch
        torch.cuda.empty_cache()


    print("Computing Eigenvalues...")
    H_cpu = hessian_sum.cpu().numpy()
    

    del hessian_sum
    gc.collect()
    torch.cuda.empty_cache()


    eigenvalue, eigenvector = np.linalg.eigh(H_cpu)

    return H_cpu, np.real(eigenvalue), np.real(eigenvector).T



def cal_hessian_minibatch_gpu(
    model, data_x, data_y, layer_index, loss_fn, batch_size=64
):
    device = next(model.parameters()).device
    model = copy.deepcopy(model).to(device)

    layer = layer_index[0]
    param = list(model.parameters())[layer]

    P = param.numel() 
    hessian_sum = torch.zeros((P, P), device=device)  

    N = len(data_x)
    num_batches = (N + batch_size - 1) // batch_size
    count=0
    for b in range(num_batches):
        count+=1
        print(f"Processing batch {b+1}/{num_batches} for Hessian computation...")
        batch_x = data_x[b * batch_size:(b + 1) * batch_size].to(device)
        batch_y = data_y[b * batch_size:(b + 1) * batch_size].to(device)

        # ---------- forward ----------
        out = model(batch_x)

        # ---------- loss ----------
        if loss_fn == "mse":
            y = nn.functional.one_hot(batch_y, num_classes=out.shape[-1]).float()
            probs = nn.functional.softmax(out, dim=-1)
            loss = nn.MSELoss()(probs, y)
        elif loss_fn == "lmse":
            y = nn.functional.one_hot(batch_y, num_classes=out.shape[-1]).float()
            loss = nn.MSELoss()(out, y)
        elif loss_fn == "cse":
            loss = nn.CrossEntropyLoss()(out, batch_y)


        grad1 = torch.autograd.grad(loss, param, create_graph=True)[0]
        grad1_flat = grad1.reshape(-1)


        rows = []
        for gi in grad1_flat:

            grad2 = torch.autograd.grad(gi, param, retain_graph=True)[0]
            rows.append(grad2.reshape(-1))

        H_B = torch.stack(rows)  # shape = (P, P)


        weight = (len(batch_x) / N)
        hessian_sum += weight * H_B  

        del loss, out, grad1, grad2, H_B, rows
        torch.cuda.empty_cache()


    H_cpu = hessian_sum.detach().cpu().numpy()
    eigenvalue,eigenvector = np.linalg.eigh(H_cpu)


    del model
    gc.collect()
    torch.cuda.empty_cache()

    return H_cpu, np.real(eigenvalue), np.real(eigenvector).T


def cal_hessian_grad(model,data_x,data_y, layer_index, loss_fn):
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    layer = layer_index[0]
    parameter = list(model.parameters())[layer]
    out_put = model(data_x)
    if loss_fn == 'mse':
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        probs = nn.functional.softmax(out_put, dim=-1)
        loss = loss_func(probs,y)
    elif loss_fn == 'lmse': 
        loss_func = nn.MSELoss()
        y = nn.functional.one_hot(data_y,num_classes=out_put.shape[-1]).float()
        loss = loss_func(out_put,y)
    elif loss_fn == 'cse':
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out_put,data_y)



    grad_1 = torch.autograd.grad(loss,parameter,create_graph=True)[0]
    hessian = []
    for grad in grad_1.view(-1):
        grad_2 = torch.autograd.grad(grad,parameter,create_graph=True)[0].view(-1)
        hessian.append(grad_2.detach().cpu().numpy())

    # h = np.array(hessian)
    # eigenvalue,eigenvector = np.linalg.eig(h)

    grad_1_vector = grad_1.detach().cpu().view(-1).numpy()

    for param in model.parameters():
        if param.grad is not None:
            param.grad.to('cpu')
        param.to('cpu')
        del param.grad
        del param

    loss.to('cpu')
    loss = None
    model = None
    del model
    del loss
    gc.collect(generation=2)
    torch.cuda.empty_cache()


    return hessian, grad_1_vector



def get_weight_num(configs, para_name):
    config = copy.deepcopy(configs)
    config[para_name] = config[para_name][0]
    model = train_model.set_model(config)
    num_weight = np.sum([np.prod(list(model.parameters())[l].shape) for l in config['layer_index']])
    return num_weight
    

def result(para_name, x_variable, dim, P):
    result = {}
    result['para_name'] = para_name
    result['para'] = x_variable
    result['loss0'] = np.zeros([len(x_variable), dim, P])
    result['loss1'] = np.zeros([len(x_variable), dim, P])
    result['sigma_w'] = np.zeros([len(x_variable), dim, P])
    result['sigma_g'] = np.zeros([len(x_variable), dim, P])
    result['mean_w'] = np.zeros([len(x_variable), dim, P])
    result['mean_g'] = np.zeros([len(x_variable), dim, P])
    result['c'] = np.zeros([len(x_variable), dim, P])
    result['hessian'] = np.zeros([len(x_variable), dim, P]);0
    result['real_loss_gap'] = np.zeros([len(x_variable), P])
    result['estimate_loss_gap'] = np.zeros([len(x_variable), P])
    result['error_gap'] = np.zeros([len(x_variable), P])
    result['L2'] = np.zeros([len(x_variable), P])
    return result

def cal_w_covar(W_holder,dimension,sample_number,count):
    W_W = np.zeros([10 * sample_number, 10 * sample_number, dimension, dimension])
    for m in range(10 * sample_number):
        for n in range(10 * sample_number):
            w_covar = 0
            for i in range(count):
                w_covar += np.outer(W_holder[i, m, :].T, W_holder[i, n, :])
            W_W[m, n, :, :] = w_covar / count
    return W_W

def cal_w_cpr(sample_number,W_W,a,b):          # compare the relative size between different sample pairs (p=p',p!=p')
    W_dia_compare = np.zeros([10 * sample_number, 10 * sample_number])
    for m in range(10 * sample_number):
        for n in range(10 * sample_number):
            W_dia_compare[m, n] = np.sum(np.abs(np.diagonal(W_W[m, n, a:b, a:b]))) / len(
                np.diagonal(W_W[m, n, a:b, a:b]))

    W_offdia_compare = (np.sum(np.sum(np.abs(W_W[:, :, a:b, a:b]), -1), -1) - W_dia_compare[m, n] * len(
        np.diagonal(W_W[m, n, a:b, a:b]))) / (len(np.diagonal(W_W[m, n, a:b, a:b])) ** 2 - len(
        np.diagonal(W_W[m, n, a:b, a:b])))

    return W_dia_compare, W_offdia_compare

def cal_wa_covar_cpr(W_holder,A_holder,count,dimension):
    W_holder = np.sum(np.squeeze(np.sum(W_holder,0)),1)
    A_holder = np.sum(np.squeeze(np.sum(np.squeeze(np.sum(A_holder,0)),1)),1)
    W_A = np.outer(W_holder, A_holder) / (count * dimension**3)
    return W_A


def cal_a_covar(A_holder,sample_number,count):
    A_A = np.zeros([10 * sample_number, 10 * sample_number, 400, 400])
    A_holder = A_holder[:, :, 0:20, 0:20].reshape(count, 10 * sample_number, 400)
    for m in range(10 * sample_number):
        for n in range(10 * sample_number):
            a_covar = 0
            for i in range(count):
                a_covar += np.outer(A_holder[i, m, 0:400].T, A_holder[i, n, 0:400])
            A_A[m, n, :, :] = a_covar / count
    return A_A

def cal_a_cpr(sample_number,A_A):          # compare the relative size between different sample pairs (p=p',p!=p')
    A_dia_compare = np.zeros([10 * sample_number, 10 * sample_number])
    for m in range(10 * sample_number):
        for n in range(10 * sample_number):
            A_dia_compare[m, n] = np.sum(np.abs(np.diagonal(A_A[m, n, :, :]))) / len(
                np.diagonal(A_A[m, n, :, :]))

    A_offdia_compare = (np.sum(np.sum(np.abs(A_A), -1), -1) - A_dia_compare[m, n] * len(
        np.diagonal(A_A[m, n, :, :]))) / (len(np.diagonal(A_A[m, n, :, :])) ** 2 - len(
        np.diagonal(A_A[m, n, :, :])))

    return A_dia_compare, A_offdia_compare
import torch

def set_config(hyper_parameter, test_size, train_size, max_epoch):
    config = {}
    config['alpha'] = 0.1
    config['lss_fn'] = 'mse' # 'mse' for regression, 'cse' for classification
    config['B'] = 50
    config['max_epoch'] = max_epoch
    config['test_size'] = test_size
    config['train_size'] = train_size
    config['rho'] = 0
    config['net_size'] = 50
    config['s'] = 1
    config['d'] = 0
    config['beta'] = 0
    config['stop_loss'] = 1e-5
    config['regulization'] = False
    config['sample_holder'] = [0,1,2,3,4,5,6,7,8,9]  ## the samples used to compute AWCH
    config['class_number'] = 10
    config['layer_index'] = [1]
    config['iter_time'] = 1
    config['evaluation'] = True
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:',config['device'])
    
    if hyper_parameter == 'alpha':
        config['alpha'] = [0.005,0.01,0.02,0.05, 0.1]
    elif hyper_parameter == 'B':
        config['B'] = [50]
        print('please set your hyper-parameter setting in set_config')
    elif hyper_parameter == 's':
        config['s'] = [4,5,6,7]
    elif hyper_parameter == 'd':
        config['d'] = [0, 0.05, 0.1, 0.2, 0.3]
    elif hyper_parameter == 'beta':
        config['beta'] = [0,5e-3,1e-2,2e-2]
        config['regulization'] = True
    elif hyper_parameter == 'rho':
        config['rho'] = [0,0.091,0.13,0.167,0.2]
    elif hyper_parameter == 'train_size':
        config['train_size'] = [400, 800, 1600, 3200]
    else:
        print('please set your hyper-parameter setting in set_config')
    return config
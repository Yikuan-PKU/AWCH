from numbers import Complex
from re import S
from sympy import N
import AWD_cuda
import utils
import train_model
import model_config
import torch
import argparse
import os
def main(max_e=100, net_size=50, n_class=10, layer_index=1):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample_number = 20
    train_size = 2000
    config = model_config.set_config('none', test_size=1000, train_size=train_size, max_epoch=max_e)
    config['B'] = 50
    config['alpha'] = 0.1
    config['lss_fn'] = 'cse'
    config['dataset'] = 'mnist'  # 'mnist'  #'cifar10'  #'fdata'
    config['model'] = 'FC_multilayer' # 'FC'
    selected_layer_index = layer_index
    config['layer_index'] = [selected_layer_index]
    config['net_size'] = net_size
    config['sample_holder'] = [i for i in range(n_class)]  # all classes
    config['hidden_sizes'] = [50,40,30,20]
    loss_fn = config['lss_fn']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = config['B']
    model, correct_train_x, correct_train_y, test_x, test_y, train_loss_holder, test_loss_holder, train_accuracy_holder, test_accuracy_holder = train_model.train(config)
    sample_holder = config['sample_holder']
    layer_indices = config['layer_index']
    lr = config['alpha']

    matrix, v, components = utils.cal_hessian_cuda(model,
                                            data_x=correct_train_x,
                                            data_y=correct_train_y,
                                            layer_index=config['layer_index'],loss_fn=config['lss_fn'])
    Covar = utils.cal_noise_covar_minibatch(model,correct_train_x,correct_train_y,layer_indices,batch_size=batch_size,loss_fn=loss_fn)


    # calculate C
    C1_dia, C2_dia, C3_dia, C1, C2, C3, C1_dia_w_dia, C1_h, H_1_d, H_2_d = AWD_cuda.cal_C_cuda(model, correct_train_x, correct_train_y, sample_holder, selected_layer_index, components, batch_size=64, sample_number=sample_number, loss_fn_name=loss_fn)


    # save_dir = f"./AWCH_data/NS{net_size}_TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"
    # save_dir = f"./AWCH_data/TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"
    save_dir = f"./AWCH_data/HS{config['hidden_sizes']}_layer{selected_layer_index}_TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save all data
    data_to_save_C = {
        'C1_dia': C1_dia,
        'C1_dia_w_dia': C1_dia_w_dia,
        'C1_h': C1_h,
        'C2_dia': C2_dia,
        'C3_dia': C3_dia,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        'C': C1 + C2 + C3,
        'H_1_d': H_1_d,
        'H_2_d': H_2_d,
        'Covar': Covar,
        'Hessian': matrix,
        'train_loss_holder': train_loss_holder,
        'test_loss_holder': test_loss_holder,
        'train_accuracy_holder': train_accuracy_holder,
        'test_accuracy_holder': test_accuracy_holder

    }
    file_name_C = f"C_epoch_{config['max_epoch']}.pt"
    save_path_C = os.path.join(save_dir, file_name_C)
    print(f"Saving C data to: {save_path_C}")
    torch.save(data_to_save_C, save_path_C, pickle_protocol=4)
    print("Data saved successfully.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model with a specified max epoch.')
    parser.add_argument('--max_e', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument('--net_size', type=int, default=50, help='Network size.')
    parser.add_argument('--n_class', type=int, default=10, help='Number of classes.')
    parser.add_argument('--layer_index', type=int, default=1, help='Layer index used for Hessian/C calculations.')
    args = parser.parse_args()
    main(args.max_e, args.net_size, args.n_class, args.layer_index)
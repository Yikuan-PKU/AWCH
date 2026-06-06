import AWD_cuda
import utils
import train_model
import model_config
import torch
import argparse
import os


def parse_layer_indices(layer_indices_arg):
    if isinstance(layer_indices_arg, (list, tuple)):
        return [int(x) for x in layer_indices_arg]
    if isinstance(layer_indices_arg, int):
        return [layer_indices_arg]
    if not isinstance(layer_indices_arg, str):
        raise ValueError("layer_indices must be int, list[int], or comma-separated string.")

    layer_indices = [int(x.strip()) for x in layer_indices_arg.split(',') if x.strip()]
    if len(layer_indices) == 0:
        raise ValueError("No valid layer indices provided.")
    return layer_indices


def main(max_e, layer_indices_arg):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample_number = 20
    train_size = 2000
    config = model_config.set_config('none', test_size=1000, train_size=train_size, max_epoch=max_e)
    config['B'] = 50
    config['alpha'] = 0.1
    config['lss_fn'] = 'mse'
    config['dataset'] = 'mnist'  # 'mnist'  #'cifar10'  #'fdata'
    config['model'] = 'FC_multilayer' # 'CNN'
    config['layer_index'] = parse_layer_indices(layer_indices_arg)
    config['hidden_sizes'] = [50,50,50,50]
    loss_fn = config['lss_fn']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = config['B']
    model, correct_train_x, correct_train_y, test_x, test_y, train_loss_holder, test_loss_holder, train_accuracy_holder, test_accuracy_holder = train_model.train(config)
    sample_holder = config['sample_holder']
    layer_index = config['layer_index']
    lr = config['alpha']

    matrix, v, components = utils.cal_hessian_cuda(model,
                                            data_x=correct_train_x,
                                            data_y=correct_train_y,
                                            layer_index=config['layer_index'],loss_fn=config['lss_fn'])
    Covar = utils.cal_noise_covar_minibatch(model,correct_train_x,correct_train_y,layer_index,batch_size=batch_size,loss_fn=loss_fn)



    # Joint H1/H2 in the same parameter space as joint Covar/Hessian.
    H_1_d, H_2_d = AWD_cuda.cal_hessian_stats_cuda_multi(
        model,
        correct_train_x,
        correct_train_y,
        components,
        layer_index,
        loss_fn,
        batch_size=12
    )

    
    data_to_save = {
        'H_1_d': H_1_d,
        'H_2_d': H_2_d,
        'layer_index': layer_index,
        'Covar': Covar,
        'Hessian': matrix,
        'train_accuracy_holder': train_accuracy_holder,
        'test_accuracy_holder': test_accuracy_holder,
        'train_loss_holder': train_loss_holder,
        'test_loss_holder': test_loss_holder
    }
    # Define the folder path
    save_dir = f"./AWCH_data/HS{config['hidden_sizes']}_layer{layer_index}_TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the filename
    file_name = f"H1_H2_epoch_{config['max_epoch']}.pt"

    # Join the directory path and filename to create the full path
    full_save_path = os.path.join(save_dir, file_name)

    # Save the data
    torch.save(data_to_save, full_save_path, pickle_protocol=4)

    print(f"File saved to: {full_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train a model with a specified max epoch.')
    parser.add_argument('--max_e', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument(
        '--layer_indices',
        type=str,
        default='1',
        help='Comma-separated layer indices, e.g. "1" or "1,2,3".'
    )
    args = parser.parse_args()
    main(args.max_e, args.layer_indices)

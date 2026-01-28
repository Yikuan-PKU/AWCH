import AWD_cuda
import utils
import train_model
import model_config
import torch
import argparse
import os
def main(max_e):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample_number = 20
    train_size = 2000
    config = model_config.set_config('none', test_size=1000, train_size=train_size, max_epoch=max_e)
    config['B'] = 128
    config['alpha'] = 0.1
    config['lss_fn'] = 'cse'
    config['dataset'] = 'cifar10'  # 'mnist'  #'cifar10' 
    config['model'] = 'CNN' # 'CNN' 'FC' 'MLP'
    config['layer_index'] = [8]
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


    h_holder, g_holder = AWD_cuda.cal_h_g_cuda(model, correct_train_x, correct_train_y, sample_holder, layer_index[0], components, sample_number, loss_fn, batch_size=50)

    data_to_save = {
        'h_holder': h_holder,
        'g_holder': g_holder,
        'train_accuracy_holder': train_accuracy_holder,
        'test_accuracy_holder': test_accuracy_holder,
        'train_loss_holder': train_loss_holder,
        'test_loss_holder': test_loss_holder,
        'matrix': matrix,
        'Covar': Covar,
        'components': components,
        
    }
    # Define the folder path
    save_dir = f"./AWCH_data/TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the filename
    file_name = f"h_g_tensors_holder_epoch_{config['max_epoch']}.pt"

    # Join the directory path and filename to create the full path
    full_save_path = os.path.join(save_dir, file_name)

    # Save the data
    torch.save(data_to_save, full_save_path, pickle_protocol=4)

    print(f"File saved to: {full_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train a model with a specified max epoch.')
    parser.add_argument('--max_e', type=int, default=200, help='Maximum number of training epochs.')
    args = parser.parse_args()
    main(args.max_e)
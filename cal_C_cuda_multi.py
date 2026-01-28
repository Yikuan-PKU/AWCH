import AWD_cuda
import utils
import train_model
import model_config
import AWD
import torch
import argparse
import os
import gc  
def main(max_e=100, net_size=50, n_class=10, run_id=0): 

    current_seed = 42 + run_id
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"--- Starting Run {run_id+1} with Seed {current_seed} ---")

    sample_number = 20
    train_size = 5000
    config = model_config.set_config('none', test_size=1000, train_size=train_size, max_epoch=max_e)
    config['B'] = 50
    config['alpha'] = 0.1
    config['lss_fn'] = 'mse'
    config['dataset'] = 'mnist'
    config['model'] = 'CNN'
    config['layer_index'] = [8]
    config['net_size'] = net_size
    config['sample_holder'] = [i for i in range(n_class)]
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
    Covar = utils.cal_noise_covar_minibatch(model, correct_train_x, correct_train_y, layer_index, batch_size=batch_size, loss_fn=loss_fn)


    C1_dia, C2_dia, C3_dia, C1, C2, C3, C1_dia_w_dia, C1_h, H_1_d, H_2_d = AWD_cuda.cal_C_cuda(model, correct_train_x, correct_train_y, sample_holder, layer_index[0], components, batch_size=64, sample_number=sample_number, loss_fn_name=loss_fn)

    save_dir = f"./AWCH_data/NS{net_size}_TrainSize{train_size}_SampleN{sample_number}_ClassN{len(config['sample_holder'])}_B{config['B']}lr{config['alpha']}_lossfn_{config['lss_fn']}_model_{config['model']}_dataset_{config['dataset']}"
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
    file_name_C = f"C_epoch_{config['max_epoch']}_run_{run_id}.pt"
    save_path_C = os.path.join(save_dir, file_name_C)
    print(f"Saving C data to: {save_path_C}")
    torch.save(data_to_save_C, save_path_C, pickle_protocol=4)
    print("Data saved successfully.")
    del model
    del matrix
    del Covar
    del data_to_save_C
    del correct_train_x
    del correct_train_y
    del components
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a model with a specified max epoch.')
    parser.add_argument('--max_e', type=int, default=200, help='Maximum number of training epochs.')
    parser.add_argument('--net_size', type=int, default=50, help='Network size.')
    parser.add_argument('--n_class', type=int, default=10, help='Number of classes.')
    parser.add_argument('--runs', type=int, default=5, help='Number of repeated runs.') 
    args = parser.parse_args()
    for i in range(args.runs):
        print(f"\n========== EXECUTION {i+1}/{args.runs} ==========")
        main(args.max_e, args.net_size, args.n_class, run_id=i) 
        print("Cleaning up memory...")
        gc.collect()             
        torch.cuda.empty_cache() 
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")  
    print("\nAll runs completed.")
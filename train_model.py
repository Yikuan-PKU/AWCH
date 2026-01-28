#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import models
import data
import utils

def set_model(config):
    net_size = config['net_size']
    s = config['s']
    d = config['d']
    if config['model'] == 'FC':
        if config['dataset'] == 'mnist':
            model = models.FC_feature(28*28,net_size,net_size,d)
        elif config['dataset'] == 'cifar10':
            model = models.FC_feature(3*32*32,net_size,net_size,d)
    elif config['model'] == 'MLP':
        if config['dataset'] == 'mnist':
            model = models.MLP_feature(28*28,net_size,50,50,d)
        elif config['dataset'] == 'cifar10':
            model = models.MLP_feature(3*32*32,net_size,50,50,d)

    elif config['model'] == 'CNN':
        if config['dataset'] == 'mnist':
            model = models.CNN(c=1)
        elif config['dataset'] == 'cifar10':
            model = models.CNN(c=3)


    # model = utils.init_weight(model,s)
    return model
    

def get_grad_per_layer(model):
    grad_info = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_info[name] = p.grad.data.norm(2).item()  # L2 norm
    return grad_info

def train(config):    
    device = config['device']
    LR = config['alpha']
    batch_size = config['B']
    EPOCH = config['max_epoch']
    test_size = config['test_size']
    sample_num  = config['train_size']
    mis_label_prob = config['rho']

    beta = config['beta']
    stop_loss = config['stop_loss']
    sample_holder = config['sample_holder']
    add_regulization = config['regulization']
    if config['dataset'] == 'mnist':
        train_x,train_y,test_x,test_y = data.sub_set_task(sample_holder,sample_num)
    elif config['dataset'] == 'fdata':
        train_x,train_y,test_x,test_y = data.sub_set_fdata_task(sample_holder,sample_num)
    elif config['dataset'] == 'cifar10':
        train_x,train_y,test_x,test_y = data.sub_set_cifar10_task(sample_holder,sample_num)
        
    train_x,train_y,identity = data.mislabel(train_x,train_y,mis_label_prob = mis_label_prob)

    correct_range = int(len(train_y)*(1-mis_label_prob))

    train_x = train_x.to(device)
    test_x = test_x[0:test_size,:].to(device)
    test_y = test_y[0:test_size].to(device)
    train_y = train_y.to(device)
    correct_train_x = train_x[0:correct_range,:]
    correct_train_y = train_y[0:correct_range] 

    
    model = set_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=LR/100)
    if 'mse' in config['lss_fn'].lower():
        loss_func = nn.MSELoss()
    elif config['lss_fn'] == 'cse':
        loss_func = nn.CrossEntropyLoss()
    
    torch_dataset = Data.TensorDataset(train_x , train_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_loss_holder = []
    test_loss_holder = []
    correct_train_accuracy_holder = []
    test_accuracy_holder =[] 

    for epoch in range(EPOCH):
        model.train()
        for step, (b_x,b_y) in enumerate(train_loader):
            out_put = model(b_x)
            if add_regulization and (epoch < 200):
                if config['lss_fn'] == 'cse':
                    loss = loss_func(out_put,b_y) + beta * utils.regularization(model,0)
                elif config['lss_fn'] == 'mse':
                    y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
                    probs = nn.functional.softmax(out_put, dim=-1)
                    loss = loss_func(probs,y) + beta * utils.regularization(model,0)
                elif config['lss_fn'] == 'lmse':
                    y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
                    loss = loss_func(out_put,y) + beta * utils.regularization(model,0)
            else:
                if config['lss_fn'] == 'cse':
                    loss = loss_func(out_put,b_y) 
                elif config['lss_fn'] == 'mse':
                    y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
                    probs = nn.functional.softmax(out_put, dim=-1)
                    loss = loss_func(probs,y)
                elif config['lss_fn'] == 'lmse':
                    y = nn.functional.one_hot(b_y,num_classes=out_put.shape[-1]).float()
                    loss = loss_func(out_put,y)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    
   
        scheduler.step()
        model.eval()
        all_loss = utils.cal_loss(model,correct_train_x,correct_train_y, loss_fn=config['lss_fn'])
        train_loss_holder.append(all_loss)

        correct_train_accuracy = utils.predict_accuracy(model,data_x = correct_train_x,data_y = correct_train_y)
        correct_train_accuracy_holder.append(correct_train_accuracy)
        
        test_accuracy = utils.predict_accuracy(model,data_x = test_x,data_y = test_y)
        test_accuracy_holder.append(test_accuracy)
        test_loss = utils.cal_loss(model, test_x, test_y, loss_fn=config['lss_fn'])
        test_loss_holder.append(test_loss)
        if (epoch%1 == 0):
            print('Epoch is |',epoch,
                  'train loss is |',all_loss,
                  'test loss is |',test_loss,
                  'train accuracy is |',correct_train_accuracy,
                  'test accuracy is |',test_accuracy)
            
        if (np.mean(correct_train_accuracy_holder[-1:-10:-1])>0.9999)&(all_loss < stop_loss):
            break


    return model, correct_train_x, correct_train_y, test_x, test_y, train_loss_holder, test_loss_holder, correct_train_accuracy_holder, test_accuracy_holder
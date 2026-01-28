import numpy as np
import torchvision
import torch
import torch.utils.data as Data
from torch.utils.data import Sampler
import torchvision.transforms as transforms

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels, n_classes_per_batch, n_samples_per_class):
        if torch.is_tensor(labels):
            self.labels = labels.cpu().numpy()  
        
        elif isinstance(labels, list) and len(labels) > 0 and torch.is_tensor(labels[0]):
            
            self.labels = np.array([label.cpu().item() if label.numel() == 1 else label.cpu().numpy() 
                                  for label in labels])
        else:
            self.labels = np.array(labels)


        self.classes = np.unique(self.labels) 
        self.n_classes = len(self.classes)
        

        self.class_indices = {}
        for c in self.classes:
            self.class_indices[c] = np.where(self.labels == c)[0]
            
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.batch_size = self.n_classes_per_batch * self.n_samples_per_class

    def __iter__(self):

        n_batches = len(self.labels) // self.batch_size
        
        for _ in range(n_batches):
            
            selected_classes = np.random.choice(self.classes, self.n_classes_per_batch, replace=False)
            batch = []
            for class_id in selected_classes:
                indices_for_class = self.class_indices[class_id]
                selected_indices = np.random.choice(indices_for_class, self.n_samples_per_class, replace=len(indices_for_class) < self.n_samples_per_class)
                batch.extend(selected_indices.tolist())
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size



def sub_set_task(label_list,sample_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_mnist = True
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    
        train=True,  
        transform = torchvision.transforms.ToTensor(),                                                      
        download = download_mnist,
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    
    
    Train_x = train_data.data.type(torch.FloatTensor).view(-1,28*28)/255.
    Train_y = train_data.targets
    Test_x = test_data.test_data.type(torch.FloatTensor).view(-1,28*28)/255.   # shape from (2000, 28, 28) to (2000, 784), value in range(0,1)
    Test_y = test_data.targets
    
    indices = torch.randperm(len(Train_x))
    Train_x = Train_x[indices]
    Train_y = Train_y[indices]
    
    train_index = []
#    for k in label_list:
    count = 0
    for i in range(len(Train_x)):
        if Train_y[i] in label_list:
            train_index.append(i)
            count = count+1
        if count >= sample_number*len(label_list):
            break
    
    test_index = []
    for i in range(len(Test_x)):
        if Test_y[i] in label_list:
            test_index.append(i)

    
    
    Train_y = np.array(Train_y)
    Train_x = np.array(Train_x)
    train_x = Train_x[np.array(train_index),:]
    train_y = Train_y[np.array(train_index)]
    
    train_x = torch.tensor(train_x).type(torch.FloatTensor)
    train_y = torch.tensor(train_y).type(torch.LongTensor)

    Test_y = np.array(Test_y)
    Test_x = np.array(Test_x)
    test_x = Test_x[np.array(test_index),:]
    test_y = Test_y[np.array(test_index)]
    
    test_x = torch.tensor(test_x).type(torch.FloatTensor)
    test_y = torch.tensor(test_y).type(torch.LongTensor)
    
    train_x = train_x.view(-1,1,28,28)
    test_x = test_x.view(-1,1,28,28)
    
    return train_x.to(device),train_y.to(device),test_x.to(device),test_y.to(device)



def sub_set_cifar10_task(label_list, sample_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_cifar = True
    
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),     
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),   
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='./cifar/',
        train=True,
        transform=transform_train,
        download=download_cifar,
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./cifar/',
        train=False,
        transform=transform_test,
        download=download_cifar,
    )

    all_train_x = []
    all_train_y = []
    for i in range(len(train_data)):
        x, y = train_data[i]   
        all_train_x.append(x)
        all_train_y.append(y)
    all_train_x = torch.stack(all_train_x)
    all_train_y = torch.tensor(all_train_y)

    all_test_x = []
    all_test_y = []
    for i in range(len(test_data)):
        x, y = test_data[i]
        all_test_x.append(x)
        all_test_y.append(y)
    all_test_x = torch.stack(all_test_x)
    all_test_y = torch.tensor(all_test_y)

    perm = torch.randperm(len(all_train_x))
    all_train_x = all_train_x[perm]
    all_train_y = all_train_y[perm]

    selected_idx = []
    count = 0
    for i in range(len(all_train_x)):
        if all_train_y[i].item() in label_list:
            selected_idx.append(i)
            count += 1
        if count >= sample_number * len(label_list):
            break

    test_idx = [i for i in range(len(all_test_x)) if all_test_y[i].item() in label_list]

    train_x = all_train_x[selected_idx]
    train_y = all_train_y[selected_idx]
    test_x = all_test_x[test_idx]
    test_y = all_test_y[test_idx]

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)



def shuffle_data_set(data_x,data_y):
    torch.manual_seed(42)
    indices = torch.randperm(len(data_x))
    data_x = data_x[indices]
    data_y = data_y[indices]
    return data_x,data_y

def mislabel(train_x,train_y,mis_label_prob):
    
    index = np.arange(0,len(train_y))
    
    train_y = train_y[index]
    mis_train_x = train_x[index,:]
    
    mis_range = int(len(train_y)*(1-mis_label_prob))
    mis_index = np.arange(mis_range,len(train_y))
    permute_index =mis_index[np.random.permutation(len(mis_index))]
    
    index = np.arange(0,len(train_y))
    index[mis_index] = index[permute_index]
    
    mis_train_y = train_y[index]
    
    for i in mis_index:
        if (mis_train_y[i] == train_y[i]):
            mis_train_y[i] = (mis_train_y[i]+np.random.randint(1,10))%10
    
    identity = np.hstack((np.ones(len(train_y)-mis_range),np.zeros(mis_range)))
    return mis_train_x,mis_train_y,identity

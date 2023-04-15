import random
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from DeepScena import DeepScena
from Network import AutoEncoder, Mutual_net, myBottleneck
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torch import optim
from torch import nn
import os

# read cell*gene preprocessed matrix
data = pd.read_csv("Data/pbmcpre.csv",header=0, index_col=0)

# read file with labels of cell types/clusters. header=None if no header, index_col=0 if have index column
csv_label = pd.read_csv("Data/celltypes.csv", header=0, index_col=None)

torch.manual_seed(100)

class read_Data(Dataset):  # inherit Dataset of torch

    def __init__(self, root_dir, transform=None):  
        self.root_dir = root_dir 
        self.transform = transform  
        self.data = self.load_data()
    def load_data(self):
        global data
        data = np.array(data).astype('float32') # convert data type
        global csv_label
        labe = np.array(csv_label)
        B = []
        for i in range(len(data)):
            t = data[i, :]
            t = t.reshape((28, 28)) # reshape each cell as 28*28
            B.append(t)
        B = np.array(B)
        labee = [int(x) for item in labe for x in item]
        data_list=[]
        for i in range(len(data)):
            data_list.append((B[i], labee[i]))
        return data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image_info, img_label = self.data[index]
        if self.transform:
            sample = self.transform(image_info)
        return sample, img_label,index # return cell data, cell cluster labels, index

batch_size = 200 # size of batch
dataset_size = data.shape[0] # number of cells in the dataset
transform_fn = transforms.Compose([transforms.ToTensor()])

train1 = read_Data(data, transform=transform_fn )

kwargs = {'num_workers': 1}


data_loader = torch.utils.data.DataLoader(
    dataset=train1,
    batch_size=batch_size,
    shuffle=True, drop_last=False, **kwargs)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

        # torch.nn.init.xavier_uniform(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
if __name__ == '__main__':

    
    num_cluster = 8 # number of cell types/clusters
    batch_size = 200
    pretraining_epoch = 0
    T1 = 2
    T2 = 1
    MaxIter1 = 20
    MaxIter2 = 20
    m = 1.5
    latent_size = 10
    zeta = 0.8
    gamma = 1 -zeta
    dataset_name = 'pbmc'
    a = 0.1
    AE = AutoEncoder(myBottleneck, [1, 1, 1]).cuda()
    AE.apply(weights_init)
    MNet = Mutual_net(num_cluster).cuda()
    MNet.apply(weights_init)
    DeepScena = DeepScena(AE,MNet,data_loader, dataset_size, batch_size=batch_size, pretraining_epoch=pretraining_epoch,
                MaxIter1=MaxIter1, MaxIter2=MaxIter2, num_cluster=num_cluster, m=m, T1=T1, T2=T2,
                latent_size=latent_size, zeta=zeta, gamma=gamma, dataset_name=dataset_name,a=a)
    
    if pretraining_epoch != 0:
        DeepScena.pretrain()
    if MaxIter1 != 0:
        DeepScena.first_module()
    if MaxIter2 != 0:
        DeepScena.second_module()

    original_label_list = []  # list of original cell labels
    latent_u_list = []  # Latent space u of first module
    latent_q_list = []  # Latent space q of second module
    predict_list = []  # predicted labels
    cell_index = []  # cell index after shuffling ( disorder)
    
    for x, target, index in data_loader:

        AE = torch.load('AE_Second_module_' + dataset_name)  
        MNet = torch.load('MNet_Second_module_' + dataset_name) 
        x = Variable(x).cuda(non_blocking=True)
        _mean, _disp, u, y = AE(x)
        q = MNet(u)
        u = u.cpu()
        q = q.cpu()
        y = torch.argmax(q, dim=1) 
        y = y.cpu()
        y = y.numpy()

        for i in range(0, x.shape[0]):
            p = index.numpy()[i]  # convert index vectors into numpy data
            cell_index.append(p)

            latent_u_list.append(u.data.numpy()[i])
            original_label_list.append(target.numpy()[i])
            predict_list.append(y[i])
            latent_q_list.append(q.data.numpy()[i])

    # orig_labels = pd.DataFrame(data=original_label_list, colums=['Original_labels'])
    # orig_labels.to_csv("original_labels.csv")
    predictedlabels = pd.DataFrame(data = predict_list, columns = ['Predicted_labels'])
    # predictedlabels.to_csv("predicted_labels.csv")
    # uspace = pd.DataFrame(data=latent_u_list)
    # uspace.to_csv("latent_u_space.csv")
    # qspace = pd.DataFrame(data=latent_q_list)
    # qspace.to_csv("latent_q_space.csv")
    cindex = pd.DataFrame(data=cell_index, index=None, columns = ['cell_index'])
    # cindex.to_csv("shuffle_index.csv")
    
    clust_result = pd.concat([cindex, predictedlabels], axis = 1)
    clust_result = clust_result.sort_values(by ="cell_index", ascending = True )
    
    clust_result.to_csv(dataset_name+'_clusters.csv', index=False)


# coding: utf-8
import sys,os
sys.path.append(os.getcwd())
import numpy as np
import random
import pickle,os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.datasets as GeoData
import argparse
from torch_geometric.data import DataLoader, DataListLoader
import torch_geometric.transforms as T

data_dir=os.path.join('DATA_PATH', 'S3DIS')

def load_S3DIS_Visual_data(obj):
    print("loading Visual data")
    Visual_dataset = VisualGraphDataset(obj)
    print("Visual no:", len(Visual_dataset))
    return Visual_dataset

class VisualGraphDataset(Dataset):
    def __init__(self,obj):
        test_file = np.load(data_dir + '/S3DIS_vis_' + str(obj)+'.npz')
        self.x = test_file['x']
        self.edge_index = test_file['edge_index']
        self.node_y = test_file['node_y']
        self.graph_y = test_file['graph_y']
        self.node_pos = test_file['node_pos']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.edge_index[index]),node_y=torch.tensor(self.node_y[index]),
             y=torch.tensor([self.graph_y[index]]),node_pos=torch.tensor(self.node_pos[index]))

def load_S3DIS_data(obj):
    print("loading train data")
    train_dataset = TrainGraphDataset(obj)
    print("train no:", len(train_dataset))
    print("loading test data")
    test_dataset = TestGraphDataset(obj)
    print("test no:", len(test_dataset))
    return train_dataset, test_dataset

# def load_S3DIS_data(obj):
#     print("loading train data")
#     train_dataset = TrainGraphDataset(obj)
#     print("train no:", len(train_dataset))
#     return train_dataset

class TrainGraphDataset(Dataset):
    def __init__(self,obj):
        train_file = np.load(data_dir+'/S3DIS_train'+str(obj)+'.npz')
        self.x=train_file['x']
        self.edge_index=train_file['edge_index']
        self.node_y=train_file['node_y']
        self.graph_y=train_file['graph_y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.edge_index[index]),node_y=torch.tensor(self.node_y[index]),
             y=torch.tensor([self.graph_y[index]]))

class TestGraphDataset(Dataset):
    def __init__(self,obj):
        test_file = np.load(data_dir + '/S3DIS_test' + str(obj) + '.npz')
        self.x = test_file['x']
        self.edge_index = test_file['edge_index']
        self.node_y = test_file['node_y']
        self.graph_y = test_file['graph_y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.edge_index[index]),node_y=torch.tensor(self.node_y[index]),
             y=torch.tensor([self.graph_y[index]]))

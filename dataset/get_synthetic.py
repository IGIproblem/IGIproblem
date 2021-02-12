# coding: utf-8
import sys,os
sys.path.append(os.getcwd())
import numpy as np
import random
import pickle,os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

data_folder=os.path.join('./dataset', 'synthetic_data')

def getUserData():
    with open(os.path.join(data_folder, "x_list.txt"), "rb") as f:
        feat = pickle.load(f)
    with open(os.path.join(data_folder, "user_label_list.txt"), "rb") as f:
        label = pickle.load(f)
    return feat,label

def get_synthetic_graph(droprate):
    with open(os.path.join(data_folder,"group_edge_index_list.txt"), "rb") as f:
        graph_edge_index = pickle.load(f)
        if droprate > 0:
            row = list(graph_edge_index[0])
            col = list(graph_edge_index[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            graph_edge_index = [row, col]
    return graph_edge_index

def load_synthetic_data():
    print("loading data set")
    data_list = GraphDataset()
    print("train no:", len(data_list))
    return data_list

class GraphDataset(Dataset):
    def __init__(self):
        with open(os.path.join(data_folder,"x_list.txt"), "rb") as f:
            self.x = pickle.load(f)
        with open(os.path.join(data_folder,"edge_index_list.txt"), "rb") as f:
            self.tuopu = pickle.load(f)
        with open(os.path.join(data_folder,"user_label_list.txt"), "rb") as f:
            self.user_y = pickle.load(f)
        with open(os.path.join(data_folder,"group_label_list.txt"), "rb") as f:
            self.group_y = pickle.load(f)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.tuopu[index]),user_y=torch.tensor(self.user_y[index]),
             y=torch.tensor([self.group_y[index]]))

def get_synthetic_train_graph(droprate):
    with open(os.path.join(data_folder,"group_edge_index_list.txt"), "rb") as f:
        graph_edge_index = pickle.load(f)
        if droprate > 0:
            row = np.array(graph_edge_index[0])
            col = np.array(graph_edge_index[1])
            res_index = []
            for i in range(len(row)):
                if row[i] < 40 and col[i] < 40:
                    res_index.append(i)
                if 50 <= row[i] < 90 and 50 <= col[i] < 90:
                    res_index.append(i)
            row = list(row[res_index])
            col = list(col[res_index])
            row=[i-10 for i in row if i>=50]
            col = [i - 10 for i in col if i >= 50]
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            graph_edge_index = [row, col]
    return graph_edge_index

def load_synthetic_train_data():
    print("loading data set")
    train_data_list = GraphTrainDataset()
    print("train no:", len(train_data_list))
    return train_data_list

class GraphTrainDataset(Dataset):
    def __init__(self):
        with open(os.path.join(data_folder,"x_list.txt"), "rb") as f:
            self.x = pickle.load(f)
        with open(os.path.join(data_folder,"edge_index_list.txt"), "rb") as f:
            self.tuopu = pickle.load(f)
        with open(os.path.join(data_folder,"user_label_list.txt"), "rb") as f:
            self.user_y = pickle.load(f)
        with open(os.path.join(data_folder,"group_label_list.txt"), "rb") as f:
            self.group_y = pickle.load(f)
        self.trainG=np.loadtxt(os.path.join(data_folder,"trainGroup.txt"))

    def __len__(self):
        return int(len(self.x)*0.8)

    def __getitem__(self, index):

        indexG=int(self.trainG[index])
        x = np.array(self.x[indexG])
        edge_index = self.tuopu[indexG]
        user_y = np.array(self.user_y[indexG])
        group_y = self.group_y[indexG]

        good_train = np.loadtxt(os.path.join(data_folder,"trainNode.txt")).astype('int64')
        good_test = np.loadtxt(os.path.join(data_folder,"testNode.txt")).astype('int64')
        train_x=x[good_train]
        train_user_y=user_y[good_train]
        newNodeID=0
        reindex={}
        for node in good_train:
            reindex[node]=newNodeID
            newNodeID+=1
        row,col=edge_index[0],edge_index[1]
        train_row,train_col=[],[]
        for i in range(len(row)):
            if row[i] in reindex.keys() and col[i] in reindex.keys():
                train_row.append(reindex[row[i]])
                train_col.append(reindex[col[i]])
        train_edge_index=[train_row,train_col]
        return Data(x=torch.FloatTensor(train_x),
                    edge_index=torch.tensor(train_edge_index), user_y=torch.tensor(train_user_y),
                    test_x=torch.FloatTensor(x),trainednodes=torch.tensor(good_train),
                    test_edge_index=torch.tensor(edge_index), test_user_y=torch.tensor(user_y),
                    y=torch.tensor([group_y]),newnodes=torch.tensor(good_test))


def load_synthetic_newnode_data(p=0.2,type='WO'):
    print("loading NewNode data set")
    data_list = NewNodeDataset(p,type)
    print("NewNode no:", len(data_list))
    return data_list

class NewNodeDataset(Dataset):
    def __init__(self,p,type):
        with open(os.path.join(data_folder,"x_list.txt"), "rb") as f:
            self.x = pickle.load(f)
        with open(os.path.join(data_folder,"edge_index_list.txt"), "rb") as f:
            self.tuopu = pickle.load(f)
        with open(os.path.join(data_folder,"user_label_list.txt"), "rb") as f:
            self.user_y = pickle.load(f)
        with open(os.path.join(data_folder,"group_label_list.txt"), "rb") as f:
            self.group_y = pickle.load(f)
        self.trainG=np.loadtxt(os.path.join(data_folder,"trainGroup.txt"))
        self.p=p
        self.type=type

    def __len__(self):
        return int(len(self.x)*0.8)

    def __getitem__(self, index):
        indexG=int(self.trainG[index])
        x = np.array(self.x[indexG])
        edge_index = self.tuopu[indexG]
        user_y = np.array(self.user_y[indexG])
        group_y = self.group_y[indexG]
        good_train = np.loadtxt(os.path.join(data_folder,"trainNode.txt"))
        good_test = np.loadtxt(os.path.join(data_folder,"testNode.txt"))
        return Data(x=torch.FloatTensor(x),
                    edge_index=torch.tensor(edge_index), user_y=torch.tensor(user_y),
                   trainednodes=torch.tensor(good_train),y=torch.tensor([group_y]),
                    newnodes=torch.tensor(good_test))

def load_synthetic_newgraph_data():

    print("loading NewGraph data set")
    data_list = NewGraphDataset()
    print("NewGraph no:", len(data_list))
    return data_list

class NewGraphDataset(Dataset):
    def __init__(self):
        with open(os.path.join(data_folder,"x_list.txt"), "rb") as f:
            self.x = pickle.load(f)
        with open(os.path.join(data_folder,"edge_index_list.txt"), "rb") as f:
            self.tuopu = pickle.load(f)
        with open(os.path.join(data_folder,"user_label_list.txt"), "rb") as f:
            self.user_y = pickle.load(f)
        with open(os.path.join(data_folder,"group_label_list.txt"), "rb") as f:
            self.group_y = pickle.load(f)
            self.testG=np.loadtxt(os.path.join(data_folder,"testGroup.txt"))

    def __len__(self):
        return int(len(self.x)*0.2)

    def __getitem__(self, index):
        indexG=int(self.testG[index])
        return Data(x=torch.FloatTensor(self.x[indexG]),
                    edge_index=torch.tensor(self.tuopu[indexG]),user_y=torch.tensor(self.user_y[indexG]),
             y=torch.tensor([self.group_y[indexG]]))


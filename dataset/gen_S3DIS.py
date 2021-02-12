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
from gcn_lib.sparse import DilatedKnnGraph
import argparse
from torch_geometric.data import DataLoader, DataListLoader
import torch_geometric.transforms as T

data_dir=os.path.join('DATA_PATH','S3DIS')

def ProcessData(obj,train):
    knn = DilatedKnnGraph(16, 1, True, 0.2)
    if train=='train':
        train_x = []
        train_edge_index = []
        train_node_y = []
        train_graph_y = []
        train_dataset = GeoData.S3DIS(data_dir, test_area=5, train=True, pre_transform=T.NormalizeScale())
        train_loader = DataLoader(train_dataset, batch_size=1 , shuffle=False, num_workers=16)
        for i, data in enumerate(train_loader):
            gt = data.y
            if obj in gt:
                num_obj = torch.sum(gt == obj)
                num_res = len(gt) - num_obj
                new_gt = [1 if label == obj else 0 for label in gt]
                train_node_y.append(new_gt)
                if num_obj >= num_res:
                    train_graph_y.append(1)
                else:
                    train_graph_y.append(0)

                corr, color, batch = data.pos, data.x, data.batch
                feat = torch.cat((corr, color), dim=1)
                train_x.append(feat.numpy())
                train_edge_index.append(knn(feat[:, 0:3],batch).numpy())

        num_1 = len([i for i,y in enumerate(train_graph_y) if y==1])
        num_0 = len([i for i,y in enumerate(train_graph_y) if y==0])
        print(num_1,num_0)
        # num_0 > num_1
        sample_index_0 = random.sample([i for i,y in enumerate(train_graph_y) if y==0], num_1)
        final_index = [i for i,y in enumerate(train_graph_y) if y==1]
        final_index.extend(sample_index_0)

        train_x = [train_x[index] for index in final_index]
        train_node_y = [train_node_y[index] for index in final_index]
        train_graph_y = [train_graph_y[index] for index in final_index]
        train_edge_index = [train_edge_index[index] for index in final_index]

        # saving training files
        np.savez(data_dir+'/S3DIS_train'+str(obj), x=np.array(train_x),
                 node_y=np.array(train_node_y),
                 graph_y=np.array(train_graph_y),
                 edge_index=np.array(train_edge_index))
    else:
        test_x = []
        test_edge_index = []
        test_node_y = []
        test_graph_y = []
        test_dataset = GeoData.S3DIS(data_dir, test_area=5, train=False, pre_transform=T.NormalizeScale())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
        for i, data in enumerate(test_loader):
            gt = data.y
            if obj in gt:
                num_obj = torch.sum(gt == obj)
                num_res = len(gt) - num_obj
                new_gt = [1 if label == obj else 0 for label in gt]
                test_node_y.append(new_gt)
                if num_obj >= num_res:
                    test_graph_y.append(1)
                else:
                    test_graph_y.append(0)

                corr, color,batch = data.pos, data.x, data.batch
                feat = torch.cat((corr, color), dim=1)
                test_x.append(feat.numpy())
                test_edge_index.append(knn(feat[:, 0:3],batch).numpy())
        # saving test files
        np.savez(data_dir+'/S3DIS_test' + str(obj), x=np.array(test_x),
                 node_y=np.array(test_node_y),
                 graph_y=np.array(test_graph_y),
                 edge_index=np.array(test_edge_index))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obj', type=int, default=0)
    parser.add_argument('--data', type=str, default='train')
    args = parser.parse_args()
    ProcessData(args.obj,args.data)

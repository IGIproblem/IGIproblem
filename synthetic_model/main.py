import sys,os
sys.path.append(os.getcwd())
import numpy as np
import argparse
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_score
from synthetic_model.GMGCN import *
from dataset.get_synthetic import get_synthetic_train_graph,load_synthetic_train_data,load_synthetic_newnode_data,load_synthetic_newgraph_data
from tools.evaluate import eva
import warnings
warnings.filterwarnings("ignore")

def train(i,epoch):
    data_loader = DataLoader(train_data_list, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2)
    train_model.train()
    optimizer.zero_grad()
    out_labels,group_x,group_y=train_model(dev,data_loader,group_edge_index)
    if args.model_name=='ATTGCN':
        loss = F.nll_loss(out_labels, group_y)
    else:
        mu=train_model.conv1.conv2.mu
        loss = total_loss(group_x, group_y, mu)
    loss.backward()
    optimizer.step()
    _, pred = out_labels.max(dim=-1)
    correct = pred.eq(group_y).sum().item()
    train_acc = correct / len(group_y)
    train_score=out_labels[:,1]
    train_auc = roc_auc_score(group_y.cpu(), train_score.cpu().detach().numpy())
    print("Iter: {:03d} | Epoch: {:05d} |  Train_Loss: {:.4f}| Train_ACC: {:.4f} | Train_AUC: {:.4f}"
          .format(i, epoch, loss.item(), train_acc, train_auc))

def node_test():
    data_loader = DataLoader(newnode_data_list, batch_size=1,
                                     shuffle=True, num_workers=2)
    test_model.eval()
    avg_trained_acc,avg_trained_F1,avg_trained_ARI,avg_trained_NMI,avg_trained_IOU_0,avg_trained_IOU_1,avg_trained_IOU = [],[],[],[], [], [],[]
    avg_new_acc, avg_new_F1, avg_new_ARI, avg_new_NMI, avg_new_IOU_0, avg_new_IOU_1, avg_new_IOU = [], [], [], [], [], [], []
    tqdm_test_loader = tqdm(data_loader)
    for Batch_data in tqdm_test_loader:
        Batch_data.to(dev)
        out = test_model(Batch_data)
        _, y_pred = out.max(dim=1)
        num_nodes=len(y_pred)
        num_pred_nodes_1 = y_pred.sum().item()
        num_pred_nodes_0 = num_nodes - num_pred_nodes_1
        if num_pred_nodes_1 >= num_pred_nodes_0 and Batch_data.y==0:
            y_pred=[1 if tem_pred==0 else 0 for tem_pred in y_pred]
            y_pred=torch.tensor(y_pred).to(dev)
        if num_pred_nodes_0 > num_pred_nodes_1 and Batch_data.y==1:
            y_pred=[1 if each_pred==0 else 0 for each_pred in y_pred]
            y_pred = torch.tensor(y_pred).to(dev)
        trainednodes = Batch_data.trainednodes.cpu().detach().numpy().astype('int64')
        newnodes = Batch_data.newnodes.cpu().detach().numpy().astype('int64')
        y,pred=Batch_data.user_y.cpu().detach().numpy(),y_pred.cpu().detach().numpy()
        trained_ACC, trained_NMI, trained_ARI, trained_F1, trained_iou_0, trained_iou_1, trained_iou = eva(y[trainednodes], pred[trainednodes])
        new_ACC, new_NMI,new_ARI,new_F1, new_iou_0, new_iou_1, new_iou = eva(y[newnodes], pred[newnodes])
        avg_trained_acc.append(trained_ACC)
        avg_new_acc.append(new_ACC)
        avg_trained_NMI.append(trained_NMI)
        avg_new_NMI.append(new_NMI)
        avg_trained_ARI.append(trained_ARI)
        avg_new_ARI.append(new_ARI)
        avg_trained_F1.append(trained_F1)
        avg_new_F1.append(new_F1)
        avg_trained_IOU.append(trained_iou)
        avg_new_IOU.append(new_iou)
        avg_trained_IOU_0.append(trained_iou_0)
        avg_new_IOU_0.append(new_iou_0)
        avg_trained_IOU_1.append(trained_iou_1)
        avg_new_IOU_1.append(new_iou_1)
    return np.mean(avg_trained_NMI),np.mean(avg_trained_acc),np.mean(avg_trained_F1),np.mean(avg_trained_ARI),np.mean(avg_trained_IOU_0),np.mean(avg_trained_IOU_1),np.mean(avg_trained_IOU),\
           np.mean(avg_new_NMI),np.mean(avg_new_acc),np.mean(avg_new_F1),np.mean(avg_new_ARI),np.mean(avg_new_IOU_0),np.mean(avg_new_IOU_1),np.mean(avg_new_IOU)

def group_test():
    data_loader = DataLoader(newgraph_data_list, batch_size=1,
                                      shuffle=True)
    test_model.eval()
    avg_acc,avg_F1,avg_ARI,avg_NMI,avg_IOU_0,avg_IOU_1,avg_IOU = [],[],[],[],[],[],[]
    tqdm_test_loader = tqdm(data_loader)
    for Batch_data in tqdm_test_loader:
        Batch_data.to(dev)
        out = test_model(Batch_data)
        _, y_pred = out.max(dim=1)
        num_nodes=len(y_pred)
        num_pred_nodes_1 = y_pred.sum().item()
        num_pred_nodes_0 = num_nodes - num_pred_nodes_1
        if num_pred_nodes_1 >= num_pred_nodes_0 and Batch_data.y==0:
            y_pred=[1 if tem_pred==0 else 0 for tem_pred in y_pred]
            y_pred=torch.tensor(y_pred).to(dev)
        if num_pred_nodes_0 > num_pred_nodes_1 and Batch_data.y==1:
            y_pred=[1 if each_pred==0 else 0 for each_pred in y_pred]
            y_pred = torch.tensor(y_pred).to(dev)
        y, pred = Batch_data.user_y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        acc, nmi, ari, f1, iou_0, iou_1, iou = eva(y, pred)
        avg_acc.append(acc)
        avg_NMI.append(nmi)
        avg_ARI.append(ari)
        avg_F1.append(f1)
        avg_IOU_0.append(iou_0)
        avg_IOU_1.append(iou_1)
        avg_IOU.append(iou)
    return np.mean(avg_NMI),np.mean(avg_acc),np.mean(avg_F1),np.mean(avg_ARI),np.mean(avg_IOU_0),np.mean(avg_IOU_1),np.mean(avg_IOU)

def main(iters):
    for epoch in range(args.n_epoch):
        train(iters,epoch)
        if epoch == args.n_epoch -1:
            trained_model = train_model.state_dict()
            trained_model["conv1.conv1.lin_l.weight"] = trained_model.pop("conv1.conv1.conv1.lin_l.weight")
            trained_model["conv1.conv1.lin_r.weight"] = trained_model.pop("conv1.conv1.conv1.lin_r.weight")
            trained_model["conv1.conv1.att_l"] = trained_model.pop("conv1.conv1.conv1.att_l")
            trained_model["conv1.conv1.att_r"] = trained_model.pop("conv1.conv1.conv1.att_r")
            trained_model["conv1.conv1.bias"] = trained_model.pop("conv1.conv1.conv1.bias")
            trained_model["conv1.conv2.lin_l.weight"] = trained_model.pop("conv1.conv1.conv2.lin_l.weight")
            trained_model["conv1.conv2.lin_r.weight"] = trained_model.pop("conv1.conv1.conv2.lin_r.weight")
            trained_model["conv1.conv2.att_l"] = trained_model.pop("conv1.conv1.conv2.att_l")
            trained_model["conv1.conv2.att_r"] = trained_model.pop("conv1.conv1.conv2.att_r")
            trained_model["conv1.conv2.bias"] = trained_model.pop("conv1.conv1.conv2.bias")
            if args.model_name=='ATTGCN':
                trained_model["pooling.b1"] = trained_model.pop("conv1.pooling.b1")
                trained_model["pooling.w1"] = trained_model.pop("conv1.pooling.w1")
                trained_model["pooling.w2"] = trained_model.pop("conv1.pooling.w2")
            else:
                trained_model["g"] = trained_model.pop("conv1.conv2.g")
                trained_model["mu"] = trained_model.pop("conv1.conv2.mu")
                trained_model["sigma"] = trained_model.pop("conv1.conv2.sigma")

            test_model.load_state_dict(trained_model, strict=False)
            trainednode_NMI, trainednode_ACC, trainednode_F1, trainednode_ARI, trainednode_IOU_0, trainednode_IOU_1, trainednode_IOU, \
            newnode_NMI, newnode_ACC, newnode_F1, newnode_ARI, newnode_IOU_0, newnode_IOU_1, newnode_IOU = node_test()
            print('Iter: {:03d} | Epoch: {:05d}| original nodes:ACC: {:.4f} | NMI: {:.4f} | ARI: {:.4f} | F1: {:.4f} | IoU_0: {:.4f} | IoU_1: {:.4f} | IoU: {:.4f} \n new nodes: ACC: {:.4f} | NMI: {:.4f} | ARI: {:.4f} | F1: {:.4f} | IoU_0: {:.4f} | IoU_1: {:.4f} | IoU: {:.4f} '
                  .format(iters, epoch, trainednode_ACC, trainednode_NMI, trainednode_ARI, trainednode_F1, trainednode_IOU_0, trainednode_IOU_1, trainednode_IOU, newnode_ACC, newnode_NMI, newnode_ARI, newnode_F1, newnode_IOU_0, newnode_IOU_1, newnode_IOU))
            newgraph_NMI, newgraph_ACC, newgraph_F1, newgraph_ARI, newgraph_IOU_0, newgraph_IOU_1, newgraph_IOU = group_test()
            print('Iter: {:03d} | Epoch: {:05d} | new graph: ACC: {:.4f} | NMI: {:.4f} | ARI: {:.4f} | F1: {:.4f} | IoU_0: {:.4f} | IoU_1: {:.4f} | IoU: {:.4f} '
                  .format(iters, epoch, newgraph_ACC, newgraph_NMI, newgraph_ARI, newgraph_F1, newgraph_IOU_0, newgraph_IOU_1, newgraph_IOU))
            return trainednode_NMI, trainednode_ACC, trainednode_F1, trainednode_ARI, trainednode_IOU_0, trainednode_IOU_1, trainednode_IOU,\
                   newnode_NMI, newnode_ACC, newnode_F1,newnode_ARI, newnode_IOU_0, newnode_IOU_1, newnode_IOU,\
                   newgraph_NMI, newgraph_ACC, newgraph_F1,newgraph_ARI, newgraph_IOU_0, newgraph_IOU_1, newgraph_IOU

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name',type=str,default='GMGCN') #ATTGCN
    parser.add_argument('--user_in_feats', type=int, default=20)
    parser.add_argument('--user_hid_feats', type=int, default=128)
    parser.add_argument('--user_out_feats', type=int, default=64)
    parser.add_argument('--group_in_feats', type=int, default=16)
    parser.add_argument('--group_hid_feats', type=int, default=64)
    parser.add_argument('--group_out_feats', type=int, default=2)
    parser.add_argument('--att_feats', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=str, default='1')
    parser.add_argument('--drop_rate', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=800)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--hier', type=str, default='W') #'WO'->without hier;'W'->with hier
    args = parser.parse_args()
    args.beta=float(args.beta)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    group_edge_index = torch.tensor(get_synthetic_train_graph(args.drop_rate)).to(dev)
    train_data_list = load_synthetic_train_data()
    newnode_data_list = load_synthetic_newnode_data(p=0.2,type=args.hier)
    newgraph_data_list =load_synthetic_newgraph_data()

    trainednode_avg_NMI, trainednode_avg_ACC, trainednode_avg_F1, trainednode_avg_ARI, trainednode_avg_IOU_0, trainednode_avg_IOU_1, trainednode_avg_IOU= [], [], [], [], [], [], []
    newnode_avg_NMI, newnode_avg_ACC, newnode_avg_F1, newnode_avg_ARI, newnode_avg_IOU_0, newnode_avg_IOU_1, newnode_avg_IOU = [], [], [], [], [], [], []
    newgraph_avg_NMI, newgraph_avg_ACC, newgraph_avg_F1, newgraph_avg_ARI, newgraph_avg_IOU_0, newgraph_avg_IOU_1, newgraph_avg_IOU = [], [], [], [], [], [], []
    for iters in range(args.num_iters):
        train_model = GMGCN(args.user_in_feats,args.user_hid_feats,args.user_out_feats,args.group_in_feats,
                            args.group_hid_feats, args.group_out_feats,args.att_feats,args.kernel_size,args.beta,
                            args.model_name,args.hier).to(dev)
        test_model = NodeInference(args.user_in_feats,args.user_hid_feats,args.user_out_feats,args.att_feats,
                                   args.kernel_size, args.group_in_feats,args.model_name).to(dev)
        total_loss = TotalLossFunc()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=args.lr)
        trainednode_NMI, trainednode_ACC, trainednode_F1, trainednode_ARI, trainednode_IOU_0, trainednode_IOU_1, trainednode_IOU, \
        newnode_NMI, newnode_ACC, newnode_F1, newnode_ARI, newnode_IOU_0, newnode_IOU_1, newnode_IOU, \
        newgraph_NMI, newgraph_ACC, newgraph_F1, newgraph_ARI, newgraph_IOU_0, newgraph_IOU_1, newgraph_IOU = main(iters)
        trainednode_avg_NMI.append(trainednode_NMI)
        trainednode_avg_ACC.append(trainednode_ACC)
        trainednode_avg_F1.append(trainednode_F1)
        trainednode_avg_ARI.append(trainednode_ARI)
        trainednode_avg_IOU_0.append(trainednode_IOU_0)
        trainednode_avg_IOU_1.append(trainednode_IOU_1)
        trainednode_avg_IOU.append(trainednode_IOU)
        newnode_avg_NMI.append(newnode_NMI)
        newnode_avg_ACC.append(newnode_ACC)
        newnode_avg_F1.append(newnode_F1)
        newnode_avg_ARI.append(newnode_ARI)
        newnode_avg_IOU_0.append(newnode_IOU_0)
        newnode_avg_IOU_1.append(newnode_IOU_1)
        newnode_avg_IOU.append(newnode_IOU)
        newgraph_avg_NMI.append(newgraph_NMI)
        newgraph_avg_ACC.append(newgraph_ACC)
        newgraph_avg_F1.append(newgraph_F1)
        newgraph_avg_ARI.append(newgraph_ARI)
        newgraph_avg_IOU_0.append(newgraph_IOU_0)
        newgraph_avg_IOU_1.append(newgraph_IOU_1)
        newgraph_avg_IOU.append(newgraph_IOU)
    print(args.model_name)
    print("original nodes:")
    print("mean ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.mean(trainednode_avg_ACC), np.mean(trainednode_avg_NMI), np.mean(trainednode_avg_ARI),
                  np.mean(trainednode_avg_F1), np.mean(trainednode_avg_IOU_0), np.mean(trainednode_avg_IOU_1), np.mean(trainednode_avg_IOU)))
    print("std ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.std(trainednode_avg_ACC), np.std(trainednode_avg_NMI), np.std(trainednode_avg_ARI),
                  np.std(trainednode_avg_F1), np.std(trainednode_avg_IOU_0), np.std(trainednode_avg_IOU_1), np.std(trainednode_avg_IOU)))

    print("new nodes:")
    print("mean ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.mean(newnode_avg_ACC), np.mean(newnode_avg_NMI), np.mean(newnode_avg_ARI),
                  np.mean(newnode_avg_F1), np.mean(newnode_avg_IOU_0), np.mean(newnode_avg_IOU_1), np.mean(newnode_avg_IOU)))
    print("std ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.std(newnode_avg_ACC), np.std(newnode_avg_NMI), np.std(newnode_avg_ARI),
                  np.std(newnode_avg_F1), np.std(newnode_avg_IOU_0), np.std(newnode_avg_IOU_1), np.std(newnode_avg_IOU)))

    print("new graph:")
    print("mean ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.mean(newgraph_avg_ACC), np.mean(newgraph_avg_NMI), np.mean(newgraph_avg_ARI),
                  np.mean(newgraph_avg_F1), np.mean(newgraph_avg_IOU_0), np.mean(newgraph_avg_IOU_1), np.mean(newgraph_avg_IOU)))
    print("std ACC:{:.5f} NMI:{:.5f} ARI:{:.5f} F1:{:.5f} IoU_0:{:.5f} IoU_1:{:.5f} IoU:{:.5f}"
          .format(np.std(newgraph_avg_ACC), np.std(newgraph_avg_NMI), np.std(newgraph_avg_ARI),
                  np.std(newgraph_avg_F1), np.std(newgraph_avg_IOU_0), np.std(newgraph_avg_IOU_1), np.std(newgraph_avg_IOU)))
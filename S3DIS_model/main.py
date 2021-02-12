import sys,os
sys.path.append(os.getcwd())
import numpy as np
import argparse
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from sklearn.metrics import roc_auc_score,precision_score
from S3DIS_model.GMGCN import *
from dataset.get_S3DIS import load_S3DIS_data,load_S3DIS_Visual_data
from tools.evaluate import eva
import warnings
import torch
import json
warnings.filterwarnings("ignore")

def train(i,epoch):
    if args.multi_gpus:
        data_loader = DataListLoader(train_data_list, batch_size=args.batch_size, shuffle=True, num_workers=16)
    else:
        data_loader = DataLoader(train_data_list, batch_size=args.batch_size,
                                   shuffle=True, num_workers=16,pin_memory=True)
    train_model1.train()
    train_model2.train()
    optimizer.zero_grad()
    for batch_idx, Batch_data in enumerate(data_loader):
        batch_x=train_model1(Batch_data)
        gt = torch.cat([data_batch.y for data_batch in Batch_data], 0).to(dev)
        if batch_idx == 0:
            group_x = batch_x
            group_y = gt
        else:
            group_x = torch.cat((group_x, batch_x), 0)
            group_y = torch.cat((group_y, gt), 0)
    group_y = group_y.squeeze(1) if group_y.dim() == 2 else group_y
    out_labels=train_model2(group_x)
    if args.multi_gpus:
        mu=train_model1.module.conv2.mu
    else:
        mu=train_model1.conv2.mu
    loss = total_loss(group_x, group_y,mu)
    loss.backward()
    optimizer.step()
    _, pred = out_labels.max(dim=-1)
    correct = pred.eq(group_y).sum().item()
    train_acc = correct / len(group_y)
    train_score=out_labels[:,1]
    train_auc=roc_auc_score(group_y.cpu(),train_score.cpu().detach().numpy())
    train_pre = precision_score(group_y.cpu(), pred.cpu())
    print( "Iter: {:03d} | Epoch: {:05d} |  Train_Loss: {:.4f}| Train_ACC: {:.4f} | Train_AUC: {:.4f}| Train_Pre: {:.4f}"
           .format(i,epoch,loss.item(),train_acc,train_auc,train_pre))


def test():
    if args.testdata == 'test':
        test_data_loader = DataLoader(test_data_list, batch_size=1,
                                          shuffle=True, num_workers=5, pin_memory=torch.cuda.is_available())
    elif args.testdata == 'train':
        test_data_loader = DataLoader(train_data_list, batch_size=1,
                                          shuffle=True, num_workers=5, pin_memory=torch.cuda.is_available())
    test_model.eval()
    avg_acc, avg_nmi, avg_ari, avg_f1,avg_iou_0,avg_iou_1,avg_iou = [], [], [], [],[],[],[]
    for i,Batch_data in enumerate(test_data_loader):
        Batch_data.to(dev)
        out = test_model(Batch_data)
        _, y_pred = out.max(dim=1)
        num_nodes=len(y_pred)
        num_pred_nodes_1 = y_pred.sum().item()
        num_pred_nodes_0 = num_nodes - num_pred_nodes_1
        if num_pred_nodes_1 > num_pred_nodes_0 and Batch_data.y==0:
            y_pred=[1 if each_pred==0 else 0 for each_pred in y_pred]
            y_pred=torch.tensor(y_pred).to(dev)
        if num_pred_nodes_0 > num_pred_nodes_1 and Batch_data.y==1:
            y_pred=[1 if each_pred==0 else 0 for each_pred in y_pred]
            y_pred = torch.tensor(y_pred).to(dev)
        y, pred = Batch_data.node_y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        acc, nmi, ari, f1, iou_0,iou_1,iou=eva(y,pred)
        avg_acc.append(acc)
        avg_nmi.append(nmi)
        avg_ari.append(ari)
        avg_f1.append(f1)
        avg_iou.append(iou)
        avg_iou_0.append(iou_0)
        avg_iou_1.append(iou_1)
    return np.mean(avg_acc),np.mean(avg_nmi),np.mean(avg_ari),np.mean(avg_f1), np.mean(avg_iou_0), np.mean(avg_iou_1), np.mean(avg_iou)


def main(iters):
    for epoch in range(args.n_epoch):
        train(iters,epoch)
        if epoch == args.n_epoch - 1:
            torch.save(train_model1.state_dict(), '/MODEL_PATH/IGI/GMGCN/'+str(args.obj)
                       +str(args.user_hid_feats)+str(args.user_out_feats)+str(args.group_in_feats)+'.pkl')
            trained_model = train_model1.state_dict()
            if args.multi_gpus:
                trained_model["conv1.conv1.att_l"] = trained_model.pop("module.conv1.conv1.att_l")
                trained_model["conv1.conv1.att_r"] = trained_model.pop("module.conv1.conv1.att_r")
                trained_model["conv1.conv1.bias"] = trained_model.pop("module.conv1.conv1.bias")
                trained_model["conv1.conv1.lin_l.weight"] = trained_model.pop("module.conv1.conv1.lin_l.weight")
                trained_model["conv1.conv1.lin_r.weight"] = trained_model.pop("module.conv1.conv1.lin_r.weight")
                trained_model["conv1.conv2.att_l"] = trained_model.pop("module.conv1.conv2.att_l")
                trained_model["conv1.conv2.att_r"] = trained_model.pop("module.conv1.conv2.att_r")
                trained_model["conv1.conv2.bias"] = trained_model.pop("module.conv1.conv2.bias")
                trained_model["conv1.conv2.lin_l.weight"] = trained_model.pop("module.conv1.conv2.lin_l.weight")
                trained_model["conv1.conv2.lin_r.weight"] = trained_model.pop("module.conv1.conv2.lin_r.weight")

                trained_model["g"] = trained_model.pop("module.conv2.g")
                trained_model["mu"] = trained_model.pop("module.conv2.mu")
                trained_model["sigma"] = trained_model.pop("module.conv2.sigma")
            else:
                trained_model["g"] = trained_model.pop("conv2.g")
                trained_model["mu"] = trained_model.pop("conv2.mu")
                trained_model["sigma"] = trained_model.pop("conv2.sigma")

            test_model.load_state_dict(trained_model, strict=False)
            acc, nmi, ari, f1, iou_0,iou_1, iou = test()
            print('Iter: {:03d}, Epoch: {:05d}, ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, mIoU_0: {:.4f}, mIoU_1: {:.4f}, mIoU: {:.4f}'
                  .format(iters, epoch, acc, nmi, ari, f1, iou_0,iou_1, iou))
            return acc, nmi, ari, f1, iou_0, iou_1, iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_name', type=str, default='S3DIS')
    parser.add_argument('--user_in_feats', type=int, default=9)
    parser.add_argument('--user_hid_feats',type=int,default=16) #32
    parser.add_argument('--user_out_feats',type=int,default=16) #16
    parser.add_argument('--group_in_feats', type=int, default=16) #16
    parser.add_argument('--group_hid_feats', type=int, default=32) #32
    parser.add_argument('--group_out_feats', type=int, default=2)
    parser.add_argument('--att_feats', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=800)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--obj', type=int, default=2)
    parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')
    parser.add_argument('--testdata', type=str, default='train')
    args = parser.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data_name == 'S3DIS':
        train_data_list, test_data_list = load_S3DIS_data(args.obj)


    avg_acc, avg_nmi, avg_ari, avg_f1, avg_iou_0 , avg_iou_1, avg_iou = [], [], [], [], [], [], []
    for iters in range(args.num_iters):
        if args.multi_gpus:
            train_model1 = DataParallel(GroupEmbedding(args.user_in_feats, args.user_hid_feats, args.user_out_feats, args.group_in_feats,
                               args.att_feats, args.kernel_size).to(dev))
        else:
            train_model1 = GroupEmbedding(args.user_in_feats,args.user_hid_feats,args.user_out_feats,args.group_in_feats, args.att_feats,args.kernel_size).to(dev)
        train_model2 = hierGCN(args.group_in_feats, args.group_out_feats).to(dev)
        test_model = NodeInference(args.user_in_feats,args.user_hid_feats,args.user_out_feats, args.kernel_size, args.group_in_feats).to(dev)
        total_loss = TotalLossFunc()
        optimizer = torch.optim.Adam([{'params':train_model1.parameters()},{'params':train_model2.parameters()}], lr=args.lr)
        acc, nmi, ari, f1, iou_0, iou_1, iou = main(iters)
        avg_acc.append(acc)
        avg_nmi.append(nmi)
        avg_ari.append(ari)
        avg_f1.append(f1)
        avg_iou_0.append(iou_0)
        avg_iou_1.append(iou_1)
        avg_iou.append(iou)
    print(args.data_name)
    print('{:03d} Iters: mean | ACC: {:.4f} | NMI: {:.4f} | ARI: {:.4f} | F1: {:.4f} | mIoU_0: {:.4f} | mIoU_1: {:.4f} | mIoU: {:.4f}'
          .format(iters, np.mean(avg_acc), np.mean(avg_nmi), np.mean(avg_ari), np.mean(avg_f1), np.mean(avg_iou_0), np.mean(avg_iou_1), np.mean(avg_iou)))
    print('{:03d} Iters: std | ACC: {:.4f} | NMI: {:.4f} | ARI: {:.4f} | F1: {:.4f} | mIoU_0: {:.4f} | mIoU_1: {:.4f} | mIoU: {:.4f}'
          .format(iters, np.std(avg_acc), np.std(avg_nmi), np.std(avg_ari), np.std(avg_f1), np.std(avg_iou_0), np.std(avg_iou_1), np.std(avg_iou)))
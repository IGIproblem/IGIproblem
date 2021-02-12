import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from synthetic_model.hierGATConv import hierGATConv
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from tqdm import tqdm
import copy
from gcn_lib.sparse import DilatedKnnGraph

EPS = 1e-15
torch.manual_seed(12345)

class TotalLossFunc(torch.nn.Module):
    def __init__(self):
        super(TotalLossFunc, self).__init__()

    def m_distance(self, feat, means):
        reshape_mean = means.unsqueeze(1) #[K,1,M]
        expand_data = feat.unsqueeze(0) #[1,N,M]
        data_mins_mean = expand_data - reshape_mean #[K,N,M]
        pair_m_distance = torch.matmul(data_mins_mean,
                        data_mins_mean.transpose(1, 2))/ 2.0 #[K,N,M]·[K,M,N]= [K,N,N]
        return torch.sqrt(torch.diagonal(pair_m_distance, 0, 1, 2).t()) #[N,K]

    def forward(self, group_x, group_y, mu, alpha=0.5):
        num_classes=group_y.max().item() + 1
        distance=self.m_distance(group_x, mu)
        label_onehot = torch.eye(num_classes, device=group_y.device)[group_y]
        adjust_m_distance = distance + label_onehot * alpha * distance
        con_loss=F.cross_entropy(-adjust_m_distance,group_y)
        return con_loss

class GAT(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hid_feats, heads=8, concat=True,negative_slope=0.2)
        self.conv2 = GATConv(8*hid_feats, out_feats, heads=1, concat=False,negative_slope=0.2)

    def forward(self, x,edge_index):
        x=self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x =self.conv2(x, edge_index)
        return x

class hierGCN(torch.nn.Module):
    def __init__(self,in_feats,out_feats,beta):
        super(hierGCN, self).__init__()
        self.conv1 = hierGATConv(in_feats,in_feats)
        self.beta = beta
        self.fc = torch.nn.Linear(in_feats, out_feats)


    def forward(self, x,edge_index):
        x1 = self.beta*self.conv1(x, edge_index)+ (1-self.beta)*x
        x2 = self.beta * self.conv1(x1, edge_index) + (1-self.beta) * x
        x = self.beta * self.conv1(x2, edge_index) + (1-self.beta) * x
        h = x.clone()
        x = self.fc(x)
        return x, h

class AttnPooling(torch.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(AttnPooling, self).__init__()
        self.in_feats = in_feats
        self.w1 = Parameter(
            torch.Tensor(hid_feats,in_feats))
        self.b1 = Parameter(
            torch.Tensor(hid_feats))
        self.w2 = Parameter(
            torch.Tensor(out_feats,hid_feats))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.w1)
        zeros(self.b1)
        glorot(self.w2)

    def forward(self, H,batch,model_name='GMGCN',size=None):
        size = batch[-1].item() + 1 if size is None else size
        x=torch.tanh(torch.mm(self.w1,torch.transpose(H,1,0)))
        x=torch.mm(self.w2,x)
        S=softmax(torch.transpose(x,1,0),batch)
        if model_name=='ATTGCN':
            fin_x1 = scatter_add(S[:, 0].view(-1, 1) * H, batch, dim=0, dim_size=size)
            fin_x2 = scatter_add(S[:, 1].view(-1, 1) * H, batch, dim=0, dim_size=size)
            fin_x = torch.cat((fin_x1, fin_x2), 1)
            return fin_x,S
        else:
            fin_x = scatter_add(S * H, batch, dim=0, dim_size=size)
        return fin_x

class GML(torch.nn.Module):
    def __init__(self, user_out_feats, group_in_feats, att_feats, kernel_size, root_weight=False,
                 bias=False):
        super(GML, self).__init__()
        self.pooling = AttnPooling(group_in_feats, att_feats, 1)
        self.user_out_feats = user_out_feats
        self.group_in_feats = group_in_feats
        self.kernel_size = kernel_size
        self.g = Parameter(
            torch.Tensor(user_out_feats, group_in_feats * kernel_size))

        self.mu = Parameter(torch.Tensor(kernel_size, group_in_feats))
        self.sigma = Parameter(torch.Tensor(kernel_size, group_in_feats))

        if root_weight:
            self.root = Parameter(torch.Tensor(user_out_feats, group_in_feats))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(group_in_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.g)
        glorot(self.mu)
        glorot(self.sigma)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, data,model_name):
        x, edge_index=data.x, data.edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        N, K, F, M = x.size(0), self.kernel_size,self.user_out_feats, self.group_in_feats
        out = torch.matmul(x, self.g).view(N, K, M)  # [N,M*K]->[N,K,M]

        gaussian = -0.5 * (out -self.mu.view(1, K, M)).pow(2) #[N,K,M]
        gaussian = gaussian / (EPS + self.sigma.view(1, K, M).pow(2)) #[N,K,M]
        gaussian = torch.exp(gaussian)  # [N, K, M]

        out_gaussian=out * gaussian# [N, K, M]
        out = out_gaussian.sum(dim=-2)  # [N,K,M]->[N,M]
        if self.root is not None:
            out = out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        size = data.batch.max().item() + 1
        out = self.pooling(out, data.batch,model_name,size)
        return out

class GroupEmbedding(torch.nn.Module):
    def __init__(self,user_in_feats,user_hid_feats,user_out_feats, group_in_feats,att_feats, kernel_size,model_name='GMGCN'):
        super(GroupEmbedding, self).__init__()
        self.conv1 = GAT(user_in_feats,user_hid_feats,user_out_feats)
        if model_name=='ATTGCN':
            self.pooling = AttnPooling(user_out_feats,att_feats,2)
        else:
            self.conv2= GML(user_out_feats, group_in_feats, att_feats, kernel_size,root_weight=True,bias=True)
        self.model_name=model_name

    def forward(self, data):
        x= self.conv1(data.x,data.edge_index)
        data.x = F.dropout(x, training=self.training)
        if self.model_name=='ATTGCN':
            x,S= self.pooling(x, data.batch,self.model_name)
        else:
            x = self.conv2(data,self.model_name)
        return x

class GMGCN(torch.nn.Module):
    def __init__(self,user_in_feats,user_hid_feats,user_out_feats, group_in_feats,group_hid_feats, group_out_feats,att_feats, kernel_size,beta,model_name,type):
        super(GMGCN, self).__init__()
        self.type=type
        self.conv1 = GroupEmbedding(user_in_feats,user_hid_feats,user_out_feats, group_in_feats,att_feats, kernel_size,model_name)
        if self.type=='W':
            if model_name=='ATTGCN':
                self.conv2 = hierGCN(user_out_feats*2, group_out_feats,beta)
            else:
                self.conv2= hierGCN(group_in_feats, group_out_feats,beta)
        else:
            if model_name=='ATTGCN':
                self.fc1 = torch.nn.Linear(user_out_feats*2, group_hid_feats)
            else:
                self.fc1 = torch.nn.Linear(group_in_feats, group_out_feats)

    def forward(self,dev,train_loader,group_edge_index):
        tqdm_train_loader = tqdm(train_loader)
        batch_idx = 0
        for Batch_data in tqdm_train_loader:
            Batch_data.to(dev)
            batch_x = self.conv1(Batch_data)
            if batch_idx == 0:
                group_x = batch_x
                group_y = Batch_data.y
            else:
                group_x = torch.cat((group_x, batch_x), 0)
                group_y = torch.cat((group_y, Batch_data.y), 0)
            batch_idx = batch_idx + 1
        group_y=group_y.squeeze(1) if group_y.dim()==2 else group_y
        if self.type=='WO':
            h = group_x.clone()
            out_x = self.fc1(group_x)
        else:
            out_x, h = self.conv2(group_x, group_edge_index)
        out_labels = F.log_softmax(out_x, dim=1)
        return out_labels,h,group_y

class NodeInference(torch.nn.Module):
    def __init__(self,user_in_feats,user_hid_feats,user_out_feats,att_feats, kernel_size, out_channels,model_name='GMGCN'):
        super(NodeInference, self).__init__()
        self.conv1 = GAT(user_in_feats, user_hid_feats, user_out_feats)
        self.model_name=model_name
        if model_name=='ATTGCN':
            self.pooling = AttnPooling(user_out_feats, att_feats, 2)
        else:
            self.in_channels = user_out_feats
            self.out_channels = out_channels
            self.kernel_size = kernel_size

            self.g = Parameter(
                torch.Tensor(user_out_feats, out_channels * kernel_size))

            self.mu = Parameter(torch.Tensor(kernel_size, out_channels))
            self.sigma = Parameter(torch.Tensor(kernel_size, out_channels))

    def forward(self, data):
        x = self.conv1(data.x,data.edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if self.model_name=='ATTGCN':
            fin_x,res = self.pooling(x, data.batch,self.model_name)
        else:
            N, K,  M = x.size(0), self.kernel_size, self.out_channels
            out = torch.matmul(x, self.g).view(N, K, M)  # [N,M*K]->[N,K,M]
            res = torch.cosine_similarity(self.mu.repeat([N, 1, 1]), out, dim=-1)
        return res

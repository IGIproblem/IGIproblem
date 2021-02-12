from torch.utils.data import Dataset
from torch_geometric.data import Data
import json,os
import torch
project_folder=os.path.join('./dataset', 'PHEME_data')

def read(file_path):
    if not os.path.exists(file_path):
        return None
    f = open(file_path, 'r')
    con = f.read()
    f.close()
    return con

def get_PHEME_graph(event):
    f = open(os.path.join(project_folder, event+'Old2NewEvent.txt'), 'r')
    a = f.read()
    Old2NewEvent = eval(a)
    f.close()
    group_path = os.path.join(project_folder, event + '_group.json')
    text_content = read(group_path)
    json_content = json.loads(text_content)
    row=[]
    col=[]
    for value in json_content.values():
        if len(value)>1:
            for i in range(len(value)):
                for j in range(len(value)):
                    if j!=i and value[i] in Old2NewEvent.keys() \
                            and value[j] in Old2NewEvent.keys():
                        row.append(Old2NewEvent[value[i]])
                        col.append(Old2NewEvent[value[j]])
    group_tuopu=[row,col]
    return group_tuopu

def get_user_y(event,occur_times,uid):
    for users in open(os.path.join(project_folder,event+str(occur_times) + '_rumor_user.txt')):
        users = users.rstrip()
        users=users.split('\t')
    if uid in users:
        user_y=1
    else:
        user_y=0
    return user_y

def get_data(event,occur_times):
    x_list,tuopu_list,group_y_list,user_y_list=[],[],[],[]
    tempeid=0
    OldEventNo=0
    NewEventNo=0
    EventDic={}
    for line in open(os.path.join(project_folder, event+'.txt')):
        line=line.rstrip()
        group_y,eid,puid,uid = int(line.split('\t')[0]),int(line.split('\t')[1]),int(line.split('\t')[2]),int(line.split('\t')[3])
        feat = line.split('\t')[4]+line.split('\t')[5]+line.split('\t')[6]+line.split('\t')[7]+line.split('\t')[8]\
             +line.split('\t')[9]+line.split('\t')[10]+line.split('\t')[11]+line.split('\t')[12]+line.split('\t')[13]+line.split('\t')[14]
        read_uid_str=line.split('\t')[15]
        feat=[int(num) for num in feat]
        if tempeid!=eid:
            if tempeid!=0 and len(group_row)!=0 and len(list(group_user_feat.values()))>max(group_col)\
                    and len(list(group_user_feat.values())) > max(group_row):
                EventDic[OldEventNo]=NewEventNo
                tuopu_list.append([group_row, group_col])
                group_y_list.append(group_y)
                x_list.append(list(group_user_feat.values()))
                user_y_list.append(list(group_user_y.values()))
                NewEventNo += 1
            group_user_y = {}
            group_user_feat = {}
            group_row = []
            group_col = []
            group_user_y[uid - 1] = get_user_y(event,occur_times, read_uid_str)
            group_user_feat[uid - 1] = feat
            tempeid = eid
            OldEventNo += 1
        else:
            group_user_y[uid-1]=get_user_y(event,occur_times,read_uid_str)
            group_user_feat[uid-1]=feat
            if puid!=0:
                group_row.append(puid-1)
                group_col.append(uid-1)
    if len(group_row) != 0 and len(list(group_user_feat.values())) > max(group_col) \
            and len(list(group_user_feat.values())) > max(group_row):
        EventDic[OldEventNo] = NewEventNo
        tuopu_list.append([group_row, group_col])
        group_y_list.append(group_y)
        x_list.append(list(group_user_feat.values()))
        user_y_list.append(list(group_user_y.values()))
    return x_list,tuopu_list,group_y_list,user_y_list

def load_PHEME_data(event,occur_times):
    print("loading data set", )
    data_list = GraphDataset(event,occur_times)
    print("train no:", len(data_list))
    return data_list

class GraphDataset(Dataset):
    def __init__(self,event,occur_times):
        self.x,self.tuopu,self.group_y,self.user_y= get_data(event,occur_times)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.tuopu[index]),user_y=torch.tensor(self.user_y[index]),
             y=torch.tensor([self.group_y[index]]))

def collate_fn(data):
    return data

def get_graph_y(event):
    group_y_list=[]
    tempeid=0
    for line in open(os.path.join(project_folder, event+'.txt')):
        line=line.rstrip()
        group_y,eid= int(line.split('\t')[0]),int(line.split('\t')[1])
        if tempeid!=eid:
            tempeid=eid
            group_y_list.append(group_y)
    return group_y_list


def get_graph_cnt(event,uid,group_y):
    group_feat=[0,0]
    f = open(os.path.join(project_folder, event+'Old2NewEvent.txt'), 'r')
    a = f.read()
    Old2NewEvent = eval(a)
    f.close()
    group_path = os.path.join(project_folder, event + '_group.json')
    text_content = read(group_path)
    json_content = json.loads(text_content)
    group_y_list=get_graph_y(event)
    if uid not in json_content.keys():
        group_feat[group_y] += 1
    else:
        user_groups=json_content[uid]
        for value in user_groups:
            if value not in Old2NewEvent.keys():
                group_feat[group_y_list[value]]+=1
            else:
                group_feat[group_y_list[Old2NewEvent[value]]]+=1
    return group_feat

def get_addcnt_data(event,occur_times):
    x_list,tuopu_list,group_y_list,user_y_list=[],[],[],[]
    tempeid=0
    OldEventNo=0
    NewEventNo=0
    EventDic={}
    for line in open(os.path.join(project_folder, event+'.txt')):
        line=line.rstrip()
        group_y,eid,puid,uid = int(line.split('\t')[0]),int(line.split('\t')[1]),int(line.split('\t')[2]),int(line.split('\t')[3])
        feat = line.split('\t')[4]+line.split('\t')[5]+line.split('\t')[6]+line.split('\t')[7]+line.split('\t')[8]\
             +line.split('\t')[9]+line.split('\t')[10]+line.split('\t')[11]+line.split('\t')[12]+line.split('\t')[13]+line.split('\t')[14]
        read_uid_str=line.split('\t')[15]
        cnt_graph_feat = get_graph_cnt(event,read_uid_str,group_y)
        feat=[int(num) for num in feat]
        feat.extend(cnt_graph_feat)
        if tempeid!=eid:
            if tempeid!=0 and len(group_row)!=0 and len(list(group_user_feat.values()))>max(group_col)\
                    and len(list(group_user_feat.values())) > max(group_row):
                EventDic[OldEventNo]=NewEventNo
                tuopu_list.append([group_row, group_col])
                group_y_list.append(group_y)
                x_list.append(list(group_user_feat.values()))
                user_y_list.append(list(group_user_y.values()))
                NewEventNo += 1
            group_user_y = {}
            group_user_feat = {}
            group_row = []
            group_col = []
            group_user_y[uid - 1] = get_user_y(event,occur_times, read_uid_str)
            group_user_feat[uid - 1] = feat
            tempeid = eid
            OldEventNo += 1
        else:
            group_user_y[uid-1]=get_user_y(event,occur_times,read_uid_str)
            group_user_feat[uid-1]=feat
            if puid!=0:
                group_row.append(puid-1)
                group_col.append(uid-1)
    if len(group_row) != 0 and len(list(group_user_feat.values())) > max(group_col) \
            and len(list(group_user_feat.values())) > max(group_row):
        EventDic[OldEventNo] = NewEventNo
        tuopu_list.append([group_row, group_col])
        group_y_list.append(group_y)
        x_list.append(list(group_user_feat.values()))
        user_y_list.append(list(group_user_y.values()))
    return x_list,tuopu_list,group_y_list,user_y_list

def load_PHEME_addcnt_data(event,occur_times):
    print("loading data set", )
    data_list = GraphCntDataset(event,occur_times)
    print("train no:", len(data_list))
    return data_list

class GraphCntDataset(Dataset):
    def __init__(self,event,occur_times):
        self.x,self.tuopu,self.group_y,self.user_y= get_addcnt_data(event,occur_times)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        return Data(x=torch.FloatTensor(self.x[index]),
                    edge_index=torch.tensor(self.tuopu[index]),user_y=torch.tensor(self.user_y[index]),
             y=torch.tensor([self.group_y[index]]))

def get_user_data(event,occur_times):
    x_list, user_y_list = [], []
    group_y_dic = {}
    user_y_dic = {}
    feat_dic = {}
    for line in open(os.path.join(project_folder, event + '.txt')):
        line = line.rstrip()
        group_y, eid, uid = int(line.split('\t')[0]), int(line.split('\t')[1]), int(line.split('\t')[3])
        feat = line.split('\t')[4] + line.split('\t')[5] + line.split('\t')[6] + line.split('\t')[7] + line.split('\t')[
            8] \
               + line.split('\t')[9] + line.split('\t')[10] + line.split('\t')[11] + line.split('\t')[12] + \
               line.split('\t')[13] + line.split('\t')[14]
        read_uid_str = line.split('\t')[15]
        cnt_graph_feat = get_graph_cnt(event, read_uid_str,group_y)
        feat = [int(num) for num in feat]
        feat.extend(cnt_graph_feat)
        if eid not in user_y_dic.keys():
            user_y_dic[eid] = [get_user_y(event, occur_times, read_uid_str)]
        else:
            user_y_dic[eid].append(get_user_y(event, occur_times, read_uid_str))
        if eid not in group_y_dic.keys():
            group_y_dic[eid] = group_y
        if eid in feat_dic.keys():
            feat_dic[eid].append(feat)
        else:
            feat_dic[eid] = [feat]
    for eid in group_y_dic.keys():
        x_list.append(feat_dic[eid])
        user_y_list.append(user_y_dic[eid])
    return x_list, user_y_list




if __name__ == '__main__':
    occur_times = 4
    for event in ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting',
                  'sydneysiege']:
        x, tuopu, group_y, user_y = get_addcnt_data(event, occur_times)
        x_list, user_y_list=get_user_data(event, occur_times)
        print(len(x),len(x_list))
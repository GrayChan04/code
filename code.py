import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import choice
from math import gcd

def readData(datasetName, datasetType):
    head_list = []
    rel_list = []
    tail_list = []
    time_list = []
    time_list_tmp = []
    with open('./data/' + datasetName + '/' + datasetType + '.txt', 'r') as f:
        for line in f:
            line = line.split()
            head_list.append(int(line[0]))
            rel_list.append(int(line[1]))
            tail_list.append(int(line[2]))
            time_list_tmp.append(int(line[3]))
        # 对时刻进行处理，让时刻连续
        for i in range(len(time_list_tmp)):
            if (time_list_tmp[i] != 0) & (time_list_tmp[i] < time_list_tmp[i + 1]):
                t = gcd(time_list_tmp[i + 1], time_list_tmp[i])#得到时间间隔
                break
        for time in time_list_tmp:
            time = time // t
            time_list.append(time)
    return head_list, rel_list, tail_list, time_list

def makeTGDict(time_list, head_list, tail_list, list_len, ent_num, dim):
    time_graph_dict={}
    for time in list(set(time_list)):#去掉重复的时刻
        g = dgl.DGLGraph()
        g.add_nodes(ent_num)
        for i in range(list_len):
            if time_list[i] == time:
                g.add_edge(head_list[i], tail_list[i])
        g.ndata['feature'] = torch.rand(g.num_nodes(), dim)#随机生成node embedding
        time_graph_dict[time] = g
    return time_graph_dict

def makeDataDict(head_batch, rel_batch, tail_batch, time_batch):
    data_dict_list=[]
    for i in range(len(head_batch)):
        dict_temp = {}
        dict_temp['head_batched'] = head_batch[i]
        dict_temp['rel_batched'] = rel_batch[i]
        dict_temp['tail_batched'] = tail_batch[i]
        dict_temp['time_batched'] = time_batch[i]
        data_dict_list.append(dict_temp)
    return data_dict_list

def readTrainData(datasetName, batch_size, TDN_num, TAN_num):
    print("read data for training...")
    head_list, rel_list, tail_list, time_list = readData(datasetName = datasetName, datasetType = 'train')
    # 假设每一个数都代表实体、关系、时间,得到实体、关系、时间的总数
    ent_set = set(head_list) | set(tail_list)#所有的实体 用来构造TAN
    time_set = set(time_list)#所有的时间 用来构造TDN
    # max选出的是最大的id，因为id从0开始，所以总数为最大的id+1
    ent_num = max(ent_set) + 1
    rel_num = max(set(rel_list)) + 1
    time_num = max(time_set) + 1
    list_len = (len(head_list) // batch_size) * batch_size#每组batch_size个数据，多余的就扔掉
    # 生成时间字典，记录每个时刻的图
    #print("make [time]:graph dictionary...")
    #time_graph_dict = makeTGDict(time_list = time_list, head_list = head_list, tail_list = tail_list, list_len = list_len, dim = dim)
    # 转成数组形式，并划分成batches
    head_batch_tmp = np.asarray(head_list[:list_len]).reshape(-1, batch_size)
    rel_batch_tmp = np.asarray(rel_list[:list_len]).reshape(-1, batch_size)
    tail_batch_tmp = np.asarray(tail_list[:list_len]).reshape(-1, batch_size)
    time_batch_tmp = np.asarray(time_list[:list_len]).reshape(-1, batch_size)
    # 构造负例
    # 构造实际batch大小
    print("make data batches for training...")
    head_batch = head_batch_tmp
    rel_batch = rel_batch_tmp
    tail_batch = tail_batch_tmp
    time_batch = time_batch_tmp
    TAN_num = TAN_num // 2 * 2#为方便处理 把TAN_num变成偶数
    for i in range((TAN_num) + TDN_num):
        head_batch = np.concatenate((head_batch, head_batch_tmp), axis = 1)
        rel_batch = np.concatenate((rel_batch, rel_batch_tmp), axis = 1)
        tail_batch = np.concatenate((tail_batch, tail_batch_tmp), axis = 1)
        time_batch = np.concatenate((time_batch, time_batch_tmp), axis = 1)
    # 构造TAN
    for i in range(TAN_num // 2):#构造TAN的循环次数
        for heads in head_batch:#构造头负例
            for index in range(batch_size): #第几个head
                while(True):
                    c = choice(list(ent_set))
                    if c != heads[index]:
                        break
                heads[index + batch_size * (i + 1)] = c
        for tails in head_batch:#构造尾负例
            for index in range(batch_size): #第几个tail
                while(True):
                    c = choice(list(ent_set))
                    if c != tails[index]:
                        break
                heads[index + batch_size * (i + 1 + TAN_num // 2)] = c
    # 构造TDN
    for i in range(TDN_num):#循环次数
        for times in time_batch:#构造时间负例
            for index in range(batch_size): #第几个时间
                while(True):
                    c = choice(list(time_set))
                    if c != times[index]:
                        break
                heads[index + batch_size * (i + 1 + TAN_num)] = c
    print("make [index]:data_batched dictionary for training...")
    data_dict_list = makeDataDict(head_batch = head_batch, rel_batch = rel_batch, tail_batch = tail_batch, time_batch = time_batch)
    return ent_num, rel_num, time_num, head_list, tail_list, time_list, list_len, data_dict_list

def readTestData(datasetName, TDN_num, TAN_num):
    print("read data for testing...")
    head_list, rel_list, tail_list, time_list = readData(datasetName = datasetName, datasetType = 'test')
    # 假设每一个数都代表实体、关系、时间,得到实体、关系、时间的总数
    ent_set = set(head_list) | set(tail_list)#所有的实体
    time_set = set(time_list)#所有的时间
    # max选出的是最大的id，因为id从0开始，所以总数为最大的id+1
    ent_num = max(ent_set) + 1
    rel_num = max(set(rel_list)) + 1
    time_num = max(time_set) + 1
    list_len = len(head_list)
    # 生成时间字典，记录每个时刻的图
    #print("make [time]:graph dictionary...")
    #time_graph_dict = makeTGDict(time_list = time_list, head_list = head_list, tail_list = tail_list, list_len = list_len, dim = dim)
    # 转成数组形式，并划分成batches，每个batch第一个是正例
    head_batch_tmp = np.asarray(head_list).reshape(-1, 1)
    rel_batch_tmp = np.asarray(rel_list).reshape(-1, 1)
    tail_batch_tmp = np.asarray(tail_list).reshape(-1, 1)
    time_batch_tmp = np.asarray(time_list).reshape(-1, 1)
    # 构造负例
    # 构造batch大小
    print("make data batches for testing...")
    TAN_num = TAN_num // 2 * 2#为方便处理 把TAN_num变成偶数
    batch_size = TAN_num + TDN_num
    batch_num = head_batch_tmp.shape[0]
    padding = np.zeros((batch_num, batch_size)).astype(int)#构造填充数组
    head_batch = np.concatenate((head_batch_tmp, padding), axis = 1)
    rel_batch = np.concatenate((rel_batch_tmp, padding), axis = 1)
    tail_batch = np.concatenate((tail_batch_tmp, padding), axis = 1)
    time_batch = np.concatenate((time_batch_tmp, padding), axis = 1)
    # 构造TAN
    for i in range(TAN_num // 2):#构造TAN的循环次数
        for heads in head_batch:#构造头负例
            while(True):
                c = choice(list(ent_set))
                if c != heads[0]:
                    break
            heads[1 + i] = c
        for tails in tail_batch:#构造尾负例
            while(True):
                c = choice(list(ent_set))
                if c != tails[0]:
                    break
            tails[1 + TAN_num // 2 + i] = c
    # 构造TDN
    for i in range(TDN_num):#循环次数
        for times in time_batch:#构造时间负例
            while(True):
                c = choice(list(time_set))
                if c != heads[0]:
                    break
            times[1 + TAN_num + i] = c
    # 填补负例
    for i in range(batch_num):
        # 填补头负例
        rel_batch[i][1:TAN_num // 2 + 1] = rel_batch[i][0]
        tail_batch[i][1:TAN_num // 2 + 1] = tail_batch[i][0]
        time_batch[i][1:TAN_num // 2 + 1] = time_batch[i][0]
        # 填补尾负例
        head_batch[i][TAN_num // 2 + 1:TAN_num + 1] = head_batch[i][0]
        rel_batch[i][TAN_num // 2 + 1:TAN_num + 1] = rel_batch[i][0]
        time_batch[i][TAN_num // 2 + 1:TAN_num + 1] = time_batch[i][0]
        # 填补时间负例
        head_batch[i][TAN_num + 1:] = head_batch[i][0]
        rel_batch[i][TAN_num + 1:] = rel_batch[i][0]
        tail_batch[i][TAN_num + 1:] = tail_batch[i][0]
    print("make [index]:data_batched dictionary for testing...")
    data_dict_list = makeDataDict(head_batch = head_batch, rel_batch = rel_batch, tail_batch = tail_batch, time_batch = time_batch)
    return ent_num, rel_num, time_num, head_list, tail_list, time_list, list_len, data_dict_list

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.gcn_msg = fn.copy_src(src = 'feature', out = 'msg')
        self.gcn_reduce = fn.sum(msg = 'msg', out = 'feature')
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, g):
        g.update_all(self.gcn_msg, self.gcn_reduce)
        h = self.linear(g.ndata['feature'])
        h = self.dropout(h)
        g.ndata['feature'] = self.activation(h)
        return g

class GCN(nn.Module):
    def __init__(self, dim):
        super(GCN, self).__init__()
        self.gcn = GCNLayer(in_dim = dim, out_dim = dim)

    def forward(self, g):
        g = self.gcn(g)
        g = self.gcn(g)
        return g

class HyTE(nn.Module):
    def __init__(self, ent_num, rel_num, time_num, device, dim):
        super(HyTE, self).__init__()

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.time_num = time_num
        self.dim = dim

        self.ent_embedding = nn.Embedding(ent_num, dim)
        self.rel_embedding = nn.Embedding(rel_num, dim)
        self.norm_vector_embedding = nn.Embedding(time_num, dim)
        self.gcn = GCN(dim = dim).to(device)

        #GCN(g)
        #self.ent_embedding.weight = nn.Parameter(graph.ndata['feature'])
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        nn.init.xavier_uniform_(self.norm_vector_embedding.weight.data)

    def transform_time(self, e_or_r, norm_vect):
        norm_vect = F.normalize(norm_vect, p = 2, dim = -1)
        #整形 e_or_r
        e_or_r = e_or_r.view(-1, norm_vect.shape[0], e_or_r.shape[-1])
        norm_vect = norm_vect.view(-1, norm_vect.shape[0], norm_vect.shape[-1])
        e_or_r = e_or_r - torch.sum(norm_vect *  e_or_r, -1, True) * norm_vect
        e_or_r = e_or_r.view(-1, e_or_r.shape[-1])
        return e_or_r

    def cal(self, h, r, t):
        # 归一化处理
        h = F.normalize(h, p = 2, dim = -1)
        r = F.normalize(r, p = 2, dim = -1)
        t = F.normalize(t, p = 2, dim = -1)
        # 整形h、r、t，以r为标准
        h = h.view(-1, r.shape[0], h.shape[-1])
        t = t.view(-1, r.shape[0], t.shape[-1])
        r = r.view(-1, r.shape[0], r.shape[-1])
        score = h + r - t
        score = torch.norm(score, p = 2, dim = -1).flatten()
        return score

    def forward(self, data, graph, device):
        # 获取结点特征
        graph= graph.to(device)
        graph = self.gcn(graph)
        feature = graph.ndata['feature'].reshape(-1, self.ent_num, self.dim)
        feature = torch.mean(feature, dim = 0)
        self.ent_embedding.weight = nn.Parameter(feature)
        # 获得数据，并将ndarry转成tensor
        h_batched = torch.from_numpy(data['head_batched']).to(device)
        r_batched = torch.from_numpy(data['rel_batched']).to(device)
        t_batched = torch.from_numpy(data['tail_batched']).to(device)
        time_batched = torch.from_numpy(data['time_batched']).to(device)
        # h, r, t向量化
        # time法向量化
        h = self.ent_embedding(h_batched)
        r = self.rel_embedding(r_batched)
        t = self.ent_embedding(t_batched)
        norm_vector = self.norm_vector_embedding(time_batched)
        h = self.transform_time(h, norm_vector)
        r = self.transform_time(r, norm_vector)
        t = self.transform_time(t, norm_vector)
        score = self.cal(h, r, t)
        return score
    
    def predict(self, data, graph, device):
        score = self.forward(data, graph, device)
        return score

def marginLoss(score, margin, batch_size):
    p_score = score[:batch_size].view(-1, batch_size)
    n_score = score[batch_size:].view(-1, batch_size)
    loss = torch.max(p_score - n_score, -margin).mean() + margin
    return loss

def train(data_dict_list, graphs, model, epoch, batch_size, margin, lr, device):
    loss_function = marginLoss
    optimize_function = torch.optim.SGD(model.parameters(), lr)
    print("training...")
    model.train()
    for i in range(epoch):
        for data_batched in data_dict_list:
            optimize_function.zero_grad()
            score = model(data_batched, graphs, device)
            loss = loss_function(score, margin, batch_size)
            loss.backward()
            optimize_function.step()
        print("Epoch {:d} | loss: {:f}".format(i, loss.item()))
    print("...training over!")
    print("save model...")
    torch.save(model, "model.pth")
    print("over!")

def test(data_dict_list, graphs, model, device):
    ranks = torch.tensor([]).to(device)
    cnt_hit10 = 0
    cnt_hit3 = 0
    cnt_hit1 = 0
    print("testing...")
    model.eval()
    with torch.no_grad():
        for data in data_dict_list:
            score = model.predict(data, graphs, device)
            score_sorted, index_sorted= torch.sort(score, descending = False)#index_sort是原score的下标在新列表中的位置
            rank = torch.nonzero(index_sorted == 0) + torch.tensor([1]).to(device)#找到正例的排名
            if rank.item() == 1:
                cnt_hit1 += 1
            elif rank.item() <= 3:
                cnt_hit3 += 1
            elif rank.item() <= 10:
                cnt_hit10 += 1
            ranks = torch.cat((ranks, rank))
    print("...testing over!")
    mr = torch.mean(ranks).item()
    mrr = torch.mean(1.0 / ranks).item()
    hit10 = cnt_hit10 / len(data_dict_list)
    hit3 = cnt_hit3 / len(data_dict_list)
    hit1 = cnt_hit1 / len(data_dict_list)
    print("MR:", mr)
    print("MRR:", mrr)
    print("Hit@10:", hit10)
    print("Hit@3:", hit3)
    print("Hit@1:", hit1)

# 启用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 处理数据集
datasetName = 'GDELT'
TAN_num_train = 2
TDN_num_train = 2
TAN_num_test = 50
TDN_num_test = 50
batch_size = 32
dim = 50
ent_num_train, rel_num_train, time_num_train, head_list_train, tail_list_train, time_list_train, list_len_train, data_dict_list_train = readTrainData(datasetName = datasetName, batch_size = batch_size, TDN_num = TDN_num_train, TAN_num = TAN_num_train)
print("...train data process over!")
ent_num_test, rel_num_test, time_num_test, head_list_test, tail_list_test, time_list_test, list_len_test, data_dict_list_test = readTestData(datasetName = datasetName, TDN_num = TDN_num_test, TAN_num = TAN_num_test)
print("...test data process over!")
ent_num_total = max(ent_num_train, ent_num_test)
rel_num_total = max(rel_num_train, rel_num_test)
time_num_total = max(time_num_train, time_num_test)
# 生成时间字典，记录每个时刻的图
print("make [time]:graph dictionary for training...")
time_graph_dict_train = makeTGDict(time_list = time_list_train, head_list = head_list_train, tail_list = tail_list_train, list_len = list_len_train, ent_num = ent_num_total, dim = dim)
print("...making over!")
graph_list_train = list(time_graph_dict_train.values())
graphs_train = dgl.batch(graph_list_train[::8])#GPU内存不够，用1/8的图
print("make [time]:graph dictionary for testing...")
time_graph_dict_test = makeTGDict(time_list = time_list_test, head_list = head_list_test, tail_list = tail_list_test, list_len = list_len_test, ent_num = ent_num_total, dim = dim)
print("...making over!")
graph_list_test = list(time_graph_dict_test.values())
graphs_test = dgl.batch(graph_list_test[::8])#GPU内存不够，用1/8的图
# 设置模型参数和训练参数
epoch = 2
margin = 4
margin = torch.tensor([margin]).to(device)
lr = 0.001
model = HyTE(ent_num = ent_num_total,
             rel_num = rel_num_total,
             time_num = time_num_total,
             device = device,
             dim = dim).to(device)
# 训练
train(data_dict_list = data_dict_list_train,
      graphs = graphs_train,
      model = model,
      epoch = epoch,
      batch_size = batch_size,
      margin = margin,
      lr = lr,
      device = device)
# 测试
test(data_dict_list = data_dict_list_test,
     graphs = graphs_test,
     model = model,
     device = device)
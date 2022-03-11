from calendar import c
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import choice

def readData(datasetName, datasetType):
    head_list = []
    rel_list = []
    tail_list = []
    time_list = []
    with open('./data/' + datasetName + '/' + datasetType + '.txt', 'r') as f:
        for line in f:
            line = line.split()
            head_list.append(int(line[0]))
            rel_list.append(int(line[1]))
            tail_list.append(int(line[2]))
            time_list.append(int(line[3]))
    return head_list, rel_list, tail_list, time_list

def makeTGDict(time_list, head_list, tail_list, list_len, dim):
    time_graph_dict={}
    for time in list(set(time_list)):#去掉重复的时刻
        g = dgl.DGLGraph()
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

def readTrainData(datasetName, batch_size, dim, TDN_num, TAN_num):
    print("read data for training...")
    head_list, rel_list, tail_list, time_list = readData(datasetName = datasetName, datasetType = 'train')
    # 得到实体、关系、时间的总数
    ent_set = set(head_list) | set(tail_list)#所有的实体 用来构造TAN
    time_set = set(time_list)#所有的时间 用来构造TDN
    ent_num = len(ent_set)
    rel_num = len(set(rel_list))
    time_num = len(time_set)
    list_len = (len(head_list) // batch_size) * batch_size#每组batch_size个数据，多余的就扔掉
    # 生成时间字典，记录每个时刻的图
    print("make [time]:graph dictionary...")
    time_graph_dict = makeTGDict(time_list = time_list, head_list = head_list, tail_list = tail_list, list_len = list_len, dim = dim)
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
    print("make [index]:data_batched dictionary...")
    data_dict_list = makeDataDict(head_batch = head_batch, rel_batch = rel_batch, tail_batch = tail_batch, time_batch =time_batch)
    return ent_num, rel_num, time_num, time_graph_dict, data_dict_list

def readTestData(datasetName, dim, TDN_num, TAN_num):
    print("read data for testing...")
    head_list, rel_list, tail_list, time_list = readData(datasetName = datasetName, datasetType = 'test')
    ent_set = set(head_list) | set(tail_list)#所有的实体
    time_set = set(time_list)#所有的时间
    ent_num = len(ent_set)
    rel_num = len(set(rel_list))
    time_num = len(time_set)
    list_len = len(head_list)
    # 生成时间字典，记录每个时刻的图
    print("make [time]:graph dictionary...")
    time_graph_dict = makeTGDict(time_list = time_list, head_list = head_list, tail_list = tail_list, list_len = list_len, dim = dim)
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
    padding = np.zeros((batch_num, batch_size))#构造填充数组，
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
                c = choice(list(ent_set))
                if c != heads[0]:
                    break
            times[1 + TAN_num + i] = c
    # 填补负例
    for i in range(batch_num):
        # 填补头负例
        rel_batch[i][1:TAN_num // 2] = rel_batch[i][0]
        tail_batch[i][1:TAN_num // 2] = tail_batch[i][0]
        time_batch[i][1:TAN_num // 2] = time_batch[i][0]
        # 填补尾负例
        head_batch[i][TAN_num // 2:TAN_num] = head_batch[i][0]
        rel_batch[i][TAN_num // 2:TAN_num] = rel_batch[i][0]
        time_batch[i][TAN_num // 2:TAN_num] = time_batch[i][0]
        #填补时间负例
        head_batch[i][TAN_num:] = head_batch[i][0]
        rel_batch[i][TAN_num:] = rel_batch[i][0]
        tail_batch[i][TAN_num:] = tail_batch[i][0]
    print("make [index]:data_batched dictionary...")
    data_dict_list = makeDataDict(head_batch = head_batch, rel_batch = rel_batch, tail_batch = tail_batch, time_batch =time_batch)
    return ent_num, rel_num, time_num, time_graph_dict, data_dict_list

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.gcn_msg = fn.copy_src(src = 'feature', out = 'msg')
        self.gcn_reduce = fn.sum(msg = 'msg', out = 'feature')
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = F.tanh()

    def forward(self, g):
        g.update_all(self.gcn_msg, self.gcn_reduce)
        h = self.linear(g.ndata['feature'])
        g.ndata['feature'] = self.activation(h)
        return g

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn = GCNLayer(in_dim = 100, out_dim = 100)

    def forward(self, g):
        g = self.gcn(g)
        g = self.gcn(g)
        return g

class HyTE(nn.Module):
    def __init__(self, ent_num, rel_num, time_num, dim = 100):
        super(HyTE, self).__init__()

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.time_num = time_num

        self.ent_embedding = nn.Embedding(ent_num, dim)
        self.rel_embedding = nn.Embedding(rel_num, dim)
        self.norm_vector_embedding = nn.Embedding(time_num, dim)
        self.gcn = GCN()

        #GCN(g)
        #self.ent_embedding.weight = nn.Parameer(graph.ndata['feature'])
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        nn.init.xavier_uniform_(self.norm_vector_embedding.weight.data)

    def transform_time(self, e_or_r, norm_vect):
        norm_vect = F.normalize(norm_vect, p = 2, dim = -1)
        #reshape e_or_r
        e_or_r = e_or_r.view(-1, norm_vect.shape[0], e_or_r.shape[-1])
        norm_vect = norm_vect.view(-1, norm_vect.shape[0], norm_vect.shape[-1])
        e_or_r = e_or_r - torch.sum(norm_vect *  e_or_r, -1, True) * norm_vect
        e_or_r.view(-1, e_or_r.shape[-1])
        return 

    def cal(self, h, r, t):
        #归一化处理
        h = F.normalize(h, p = 2, dim = -1)
        r = F.normalize(r, p = 2, dim = -1)
        t = F.normalize(t, p = 2, dim = -1)
        #整形h、r、t
        h = h.view(-1, r.shape[0], h.shape[-1])
        r = r.view(-1, r.shape[0], r.shape[-1])
        t = t.view(-1, r.shape[0], t.shape[-1])
        score = h + r - t
        score = torch.norm(score, 2, -1).flatten()
        return score

    def forward(self, data, graph):
        graph = GCN(graph)
        self.ent_embedding.weight = nn.Parameer(graph.ndata['feature'])
        #获得数据
        h_batched = data['head_batched']
        r_batched = data['rel_batched']
        t_batched = data['tail_batched']
        time_batched = data['time_batched']
        #h, r, t向量化
        #time法向量化
        h = self.ent_embedding(h_batched)
        r = self.rel_embedding(r_batched)
        t = self.ent_embedding(t_batched)
        norm_vector = self.norm_vector_embedding(time_batched)
        h = self.transform_time(h, norm_vector)
        r = self.transform_time(r, norm_vector)
        t = self.transform_time(t, norm_vector)
        score = self.cal(h, r, t)
        return score

    def predict(self, data):
        score = self.forwrd(self, data)
        return score

def marginLoss(score, margin, batch_size):
    p_score = score[:batch_size].view(-1, batch_size)
    n_score = score[batch_size:].view(-1, batch_size)
    margin = torch.tensor([margin])
    loss = torch.max(p_score - n_score, -margin).mean() + margin
    return loss

def train(data_dict_list, graphs, model, epoch, batch_size, margin, lr):
    loss_function = marginLoss
    optimize_function = torch.optim.SGD(model.parameter(), lr)
    print("training...")
    for i in range(epoch):
        for data_batched in data_dict_list:
            score = model(data_batched, graphs)
            loss = loss_function(score, margin, batch_size)
            optimize_function.zero_grad()
            loss.backward()
            optimize_function.step()
        if i % 10 == 0 & i != 0:
            print("Epoch %d | loss: %f" % (i, loss.item()))
    torch.save(model, "model.pth")
    return model

def test(data_dict_list, graphs, model):
    ranks = torch.tensor([])
    cnt_hit10 = 0
    cnt_hit3 = 0
    cnt_hit1 = 0
    print("testing...")
    for data in data_dict_list:
        score = model.predict(data, graphs)
        score_sorted, index_sorted= torch.sort(score, descending = False)#index_sort是原score的下标在新列表中的位置
        rank = torch.nonzero(index_sorted == 0)#找到正例的排名
        if rank.item() == 1:
            cnt_hit1 += 1
        elif rank.item() <= 3:
            cnt_hit3 += 1
        elif rank.item() <= 10:
            cnt_hit10 += 1
        ranks = torch.cat((ranks, rank))
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


#处理数据集
datasetName = 'GDELT'
TAN_num_train = 4
TDN_num_train = 4
batch_size = 1024
ent_num_train, rel_num_train, time_num_train, time_graph_dict_train, data_dict_list_train = readTrainData(datasetName = datasetName, batch_size = batch_size, dim = 100, TDN_num = TDN_num_train, TAN_num = TAN_num_train)
graph_list_train = list(time_graph_dict_train.values())
graphs_train = dgl.batch(graph_list_train)
print("train data process over")
TAN_num_test = 200
TDN_num_test = 200
ent_num_test, rel_num_test, time_num_test, time_graph_dict_test, data_dict_list_test = readTestData(datasetName = datasetName, dim = 100, TDN_num = TDN_num_test, TAN_num = TAN_num_test)
graph_list_test = list(time_graph_dict_test.values())
graphs_test = dgl.batch(graph_list_test)
print("test data process over")
#设置训练参数
epoch = 1
margin = 4
lr = 0.003
TDN_num = 200
TAN_num = 200
dim = 100
model = HyTE(ent_num = ent_num_train, rel_num = rel_num_train, time_num = time_num_train, dim = 100)
#训练
train(data_dict_list = data_dict_list_train,
      graphs = graphs_train,
      model = model,
      epoch = epoch,
      batch_size = batch_size,
      margin = margin,
      lr = lr)
#测试
test(data_dict_list = data_dict_list_test,
      graphs = graphs_test,
      model = model)

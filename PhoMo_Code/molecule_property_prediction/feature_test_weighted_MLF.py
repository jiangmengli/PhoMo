import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import json
# from utils import sparse_mx_to_torch_sparse_tensor
# from node.dataset import load
# from GAT_layers import GraphAttentionLayer, SpGraphAttentionLayer
# import torch.nn.functional as F
# from circleloss_original import CircleLoss
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import argparse

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class Model(nn.Module):
    def __init__(self, n_in, n_h, k=0.5, dropout=0.6, nheads=1, alpha=0.2):
        super(Model, self).__init__()

        super(Model, self).__init__()
        self.gcn_adj_layer1 = GCN(n_in, n_h)
        self.gcn_diff_layer1 = GCN(n_in, n_h)

        self.gcn_adj_layer2 = GCN(n_h,n_h)
        self.gcn_diff_layer2 = GCN(n_h,n_h)

    def forward(self, seq_pos, seq_neg, adj, diff, sparse, msk, samp_bias1, samp_bias2): # GH: seq1=features ,seq2=shuffled features

        #layer1
        h_adj_layer1_pos = self.gcn_adj_layer1(seq_pos, adj)
        h_diff_layer1_pos = self.gcn_diff_layer1(seq_pos, diff)

        h_adj_layer1_neg = self.gcn_adj_layer1(seq_neg, adj)
        h_diff_layer1_neg = self.gcn_diff_layer1(seq_neg, diff)

        #layer2
        h_adj_layer2_pos = self.gcn_adj_layer2(h_adj_layer1_pos, adj)
        h_diff_layer2_pos = self.gcn_diff_layer2(h_diff_layer1_pos, diff)

        h_adj_layer2_neg = self.gcn_adj_layer2(h_adj_layer1_neg, adj)
        h_diff_layer2_neg = self.gcn_diff_layer2(h_diff_layer1_neg, diff)


        return h_adj_layer1_pos, h_diff_layer1_pos, h_adj_layer2_pos, h_diff_layer2_pos, \
               h_adj_layer1_neg, h_diff_layer1_neg, h_adj_layer2_neg, h_diff_layer2_neg



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc_add = nn.Linear(ft_in, ft_in)
        self.fc_add2 = nn.Linear(ft_in, ft_in)
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        seq = self.fc_add(seq)
        seq = self.fc_add2(seq)
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret

def load_features(path, format = 'json'):
    if format == 'json':
        with open(path, 'r') as load_f:
            load_dict = json.load(load_f)
        print("loading json file:" + path)
        features = load_dict
    elif format == 'npy':
        features = np.load(f'{path}')
    else:
        raise NotImplementedError

    return features


def train(feature_gnn_path, class_num = 'multi', predition='Photosensitizer', matching_path='graph_match_judge.npy', MLF= True, sample_total_num=364):

    # feature_gnn1 = load_features("./graphormer_output_np101-206.npy", format = 'npy')
    # feature_gnn2 = load_features("./graphormer_output_np0-100.npy", format='npy')
    # feature_gnn = np.concatenate([feature_gnn1,feature_gnn2],axis=0)



    predition = 'Photosensitizer'

    feature_gnn = load_features(feature_gnn_path, format='npy')

    feature_gnn = torch.FloatTensor(feature_gnn).cuda()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    feature_exps = np.zeros((sample_total_num,10),dtype=float)
    labels_ground = np.zeros((sample_total_num),dtype=int)
    labels1 = np.zeros((sample_total_num), dtype=int)
    labels2 = np.zeros((sample_total_num), dtype=int)
    labels3 = np.zeros((sample_total_num), dtype=int)
    labels4 = np.zeros((sample_total_num), dtype=int)

    feature_attribute = load_features("./PC_experiment_json.json", format = 'json')
    feature_graph_matching = load_features(matching_path, format='npy')
    feature_graph_matching = np.round(feature_graph_matching , 1)
    feature_graph_matching = torch.FloatTensor(feature_graph_matching).cuda()
    print("feature_graph_matching is: \n", feature_graph_matching)
    counter = 0
    set_random_seed(16)

    for item in feature_attribute:
        feature_exps[counter][1] = item['Double Metal Core']
        feature_exps[counter][2] = item['Metal Core Type']
        feature_exps[counter][3] = item['Electron Donating Group']
        feature_exps[counter][4] = item['Electron Withdrawing Group']
        feature_exps[counter][5] = item['Proton Donor']

        if not 'Photosensitizer' in item:
            break

        if item['Photosensitizer'] == '-':
            feature_exps[counter][6] = 0
            feature_exps[counter][7] = 0

        if item['Photosensitizer'] == 'Ir(ppy)3':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 1
        if item['Photosensitizer'] == 'fac-Ir(ppy)3':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 2
        if item['Photosensitizer'] == 'Ir(dFCF3ppy)2(dtbbpy)':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 3
        if item['Photosensitizer'] == 'IrPs':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 4
        if item['Photosensitizer'] == '{Ir(C^N)2[cis-chelating bis(N-heterocyclic carbene)]}':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 5
        if item['Photosensitizer'] == 'Ru(ppy)3':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 6
        if item['Photosensitizer'] == 'Ru(bpy)3':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 7
        if item['Photosensitizer'] == '[Ru(bpy)3]Cl2.6H2O':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 8
        if item['Photosensitizer'] == '[Ru(bpy)3]Cl2':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 9
        if item['Photosensitizer'] == '[Ru(bpy)3](NO3)2':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 10
        if item['Photosensitizer'] == '[Ru(phen)3]':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 11
        if item['Photosensitizer'] == 'Ru':
            feature_exps[counter][6] = 1
            feature_exps[counter][7] = 12

        if item['Photosensitizer'] == 'Cu1':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 1
        if item['Photosensitizer'] == 'Cu2':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 2
        if item['Photosensitizer'] == 'CuPS':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 3
        if item['Photosensitizer'] == 'CuPP':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 4
        if item['Photosensitizer'] == '[Zn(TMPyP)]Cl4':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 5
        if item['Photosensitizer'] == 'ZnTPP':
            feature_exps[counter][6] = 2
            feature_exps[counter][7] = 6

        if item['Photosensitizer'] == '4CzIPN':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 1
        if item['Photosensitizer'] == 'p-terphenyl':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 2
        if item['Photosensitizer'] == '9CNA':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 3
        if item['Photosensitizer'] == 'Purpurin':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 4
        if item['Photosensitizer'] == 'Pheno':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 5
        if item['Photosensitizer'] == 'Phen2':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 6
        if item['Photosensitizer'] == 'Acr':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 7
        if item['Photosensitizer'] == 'phenazine':
            feature_exps[counter][6] = 3
            feature_exps[counter][7] = 8

        if item['Solution Additives_M'] == '-':
            feature_exps[counter][8] = 0
            feature_exps[counter][9] = 0
        if item['Solution Additives_M'] == 'TFE':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 1
        if item['Solution Additives_M'] == 'BIH':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 2
        if item['Solution Additives_M'] == 'TEA':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 3
        if item['Solution Additives_M'] == 'H2O':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 4
        if item['Solution Additives_M'] == 'PhOH':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 5
        if item['Solution Additives_M'] == '4-Cl-PhOH':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 6
        if item['Solution Additives_M'] == 'MeOH':
            feature_exps[counter][8] = 1
            feature_exps[counter][9] = 7
        if item['Solution Additives_M'] == 'NaHCO3':
            feature_exps[counter][8] = 2
            feature_exps[counter][9] = 8

        # TON
        if item['CO'] == '-':
            CO = 0
        else:
            CO = item['CO']
        if item['CH4'] == '-':
            CH4 = 0
        else:
            CH4 = item['CH4']
        if item['HCOOH'] == '-':
            HCOOH = 0
        elif item['HCOOH'] == '':
            HCOOH = 0
        else:
            HCOOH = item['HCOOH']
        sum_TON = float(CO) + float(CH4) + float(HCOOH)
        if sum_TON <= 50:
            labels1[counter] = 0
        if 50< sum_TON <= 500:
            labels1[counter] = 1
        if sum_TON > 500:
            labels1[counter] = 2

        # CAC
        if item['Catalyst Concentration_uM'] == '-' or item['Catalyst Concentration_uM'] == '':
            C = 0.0
        else:
            C = float(item['Catalyst Concentration_uM'])
        if C <= 1:
            labels2[counter] = 0
        if C > 1 and C <= 10:
            labels2[counter] = 1
        if C > 10 and C <= 100:
            labels2[counter] = 2
        if C > 100 and C <= 1000:
            labels2[counter] = 3
        if C > 1000:
            labels2[counter] = 4

        # if predition == "Photosensitizer":
        labels3[counter] = feature_exps[counter][6]

        # if predition == "Solution Additives_M":
        labels4[counter] = feature_exps[counter][9]

        # if predition == "cuihuaji":
        # if item['Evaluation'] ==0:
        #     labels_ground[counter] = 0
        # else:
        #     labels_ground[counter] = 1

        labels_ground[counter] = item['Evaluation']

        # print("label is", labels[counter])

        counter = counter + 1




    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    feature_exps = torch.FloatTensor(feature_exps).cuda()
    # idx_train = np.arange(100)
    # idx_test = np.arange(101,334)



    # classes number
    labels1 = torch.LongTensor(labels1).cuda() # TON
    labels2 = torch.LongTensor(labels2).cuda() # CAC
    labels3 = torch.LongTensor(labels3).cuda() # Photosensitizer
    labels4 = torch.LongTensor(labels4).cuda()
    labels_ground = torch.LongTensor(labels_ground).cuda()
    if class_num == 'binary':
        labels1 = torch.where(labels1>0, 1,0)
        labels2 = torch.where(labels2 > 0, 1, 0)
        labels3 = torch.where(labels3 > 0, 1, 0)
        labels4 = torch.where(labels4 > 0, 1, 0)
        labels_ground = torch.where(labels_ground > 0, 1, 0)
    else:
        pass
    idx_train = np.linspace(0, 333, num=334, endpoint=True, dtype=int)
    idx_test = np.linspace(0, sample_total_num-1, num=sample_total_num, endpoint=True, dtype=int)
    idx_train = torch.LongTensor(idx_train).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()

    xent = nn.CrossEntropyLoss()

    feature_zeros = np.zeros((364, 8), dtype=float)
    feature_zeros = torch.FloatTensor(feature_zeros).cuda()

    if MLF:
        embeds = torch.cat((feature_exps[:,1:6], feature_gnn, feature_graph_matching), 1)
    else:
        embeds = torch.cat((feature_exps[:,6:10], feature_gnn), 1)

    if predition == "Photosensitizer" or predition == "Solution Additives_M":
        if MLF:
            embeds = torch.cat((feature_exps[:, 0:5], feature_gnn, feature_graph_matching), 1)
        else:
            embeds = feature_gnn

    # embeds = feature_gnn
    # embeds = feature_exps
    # embeds = feature_zeros

    # embeds = embeds.squeeze()
    log_len = embeds.shape[1]
    train_embs = embeds[idx_train, :]
    test_embs = embeds[idx_test, :]

    FYTJ = np.zeros((30,4), dtype=int)

    # Predict 1
    train_lbls = labels1[idx_train]
    test_lbls = labels1[idx_test]
    #
    accs = []

    acc_graphs = []

    # for count1 in range(50):
    log = LogReg(log_len , 4) # GH: feature size , class num
    if  predition == "CAC":
        log = LogReg(log_len, 5)
    if predition == "Photosensitizer":
        log = LogReg(log_len, 15)
    if predition == "Solution Additives_M":
        log = LogReg(log_len, 15)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
    log.cuda()
    for count2 in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    logits_print = logits[-30:]
    probability_print = F.softmax(logits_print, dim=1)

    index = preds[-30:]
    counter = 0
    for label in index:
        # print("label is", index[counter])
        # print("logit all is", logits_print[counter])
        # print("probability all is", probability_print[counter])
        # print("probability is", probability_print[counter,index[counter]])
        if label == 0:
            FYTJ[counter][0] = 25
        if label == 1:
            FYTJ[counter][0] = 225
        if label == 2:
            FYTJ[counter][0] = 1000
        counter = counter + 1


    # Predict 2
    train_lbls = labels2[idx_train]
    test_lbls = labels2[idx_test]
    #
    accs = []

    acc_graphs = []

    # for count1 in range(50):
    log = LogReg(log_len , 4) # GH: feature size , class num
    if  predition == "CAC":
        log = LogReg(log_len, 5)
    if predition == "Photosensitizer":
        log = LogReg(log_len, 15)
    if predition == "Solution Additives_M":
        log = LogReg(log_len, 15)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
    log.cuda()
    for count2 in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    logits_print = logits[-30:]
    probability_print = F.softmax(logits_print, dim=1)

    index = preds[-30:]
    counter = 0
    for label in index:
        # print("label is", index[counter])
        # # print("logit all is", logits_print[counter])
        # # print("probability all is", probability_print[counter])
        # print("probability is", probability_print[counter,index[counter]])
        if label == 0:
            FYTJ[counter][1] = 5
        if label == 1:
            FYTJ[counter][1] = 55
        if label == 2:
            FYTJ[counter][1] = 550
        if label == 4:
            FYTJ[counter][1] = 10000
        counter = counter + 1


    # Predict 3

    #==============================================================
    #==================                      ======================
    #==================        RESULT        ======================
    #==================                      ======================
    #==============================================================

    weight_rank_sum = torch.zeros(30, 1).cuda()

    for itr in range(5):

        train_lbls = labels3[idx_train]
        train_lbls = train_lbls[0:227]
        test_lbls = labels3[idx_test]
        #
        accs = []

        acc_graphs = []

        # for count1 in range(50):
        log = LogReg(log_len , 4) # GH: feature size , class num
        if  predition == "CAC":
            log = LogReg(log_len, 5)
        if predition == "Photosensitizer":
            log = LogReg(log_len, 15)
        if predition == "Solution Additives_M":
            log = LogReg(log_len, 15)
        log = LogReg(log_len, 4)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
        log.cuda()
        for count2 in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs[0:227,:])
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()


        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        weight = F.softmax(logits, dim=1).cuda()  # 确保在 GPU 上
        label_w = torch.tensor([0, 0.1, 0.2, 0.3], dtype=torch.float32).cuda()  # 确保在 GPU 上

        # 确保两个张量在同一设备后再进行矩阵乘法
        weight = torch.matmul(weight, label_w.view(-1, 1))
        # 获取后30个的weight，4分类
        weight_4 = weight[-30:]

        train_lbls[train_lbls > 1] = 1
        log = LogReg(log_len, 2)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
        log.cuda()
        for count2 in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs[0:227,:])
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        weight = F.softmax(logits, dim=1).cuda()  # 确保在 GPU 上
        label_w = torch.tensor([0, 1], dtype=torch.float32).cuda()  # 确保在 GPU 上

        # 确保两个张量在同一设备后再进行矩阵乘法
        weight = torch.matmul(weight, label_w.view(-1, 1))
        # 获取后30个的weight，2分类
        weight_2 = weight[-30:]

        # 确保操作在正确的设备上
        device = weight_2.device

        # 减去最小值
        weight_2_min = weight_2.min()
        weight_2_normalized = weight_2 - weight_2_min

        # 归一化（除以最大值和最小值的差值）
        weight_2_range = weight_2_normalized.max()  # 获取最大值
        if weight_2_range > 0:  # 避免除以零
            weight_2_normalized = weight_2_normalized / weight_2_range

        # weight_2_transformed = weight_2_normalized ** 2 + 0.1 * weight_4

        weight_2_transformed = weight_2_normalized

        # weight_2 = weight_2_normalized

        weight_rank = weight_2_transformed
        weight_rank_sum = weight_rank_sum + weight_rank

        # weight_rank = weight_2_u
        # weight_rank_sum = weight_rank_sum + weight_rank


    # 确保所有张量在同一设备上
    weight_rank_sum = weight_rank_sum / 5
    weight_rank_sum = torch.sigmoid(weight_rank_sum)
    device = weight_rank_sum.device  # 获取 weight_rank 所在的设备


    # 将 original_indices 移动到与 weight_rank 相同的设备
    original_indices = torch.arange(weight_rank_sum.size(0), device=device).view(-1, 1)

    # 合并张量
    tensor_with_indices = torch.cat((weight_rank_sum, original_indices), dim=1)

    # 按值排序
    sorted_tensor = tensor_with_indices[tensor_with_indices[:, 0].argsort()]

    first_column = sorted_tensor[:, 0]
    min_val = first_column.min()
    max_val = first_column.max()

    # Normalization formula (x - min) / (max - min)
    normalized_first_column = (first_column - min_val) / (max_val - min_val)

    # Replacing the first column with the normalized values
    sorted_tensor[:, 0] = normalized_first_column

    return sorted_tensor



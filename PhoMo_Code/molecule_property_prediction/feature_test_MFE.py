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
import data_generate.create_csv_gz as create_data
import os
import numpy as np
import feature_get
import graph_search_get
import torch
import json
import csv
import both_get
import torch
import torch.nn as nn
import feature_test
import graph_search_get
import torch
import graph_search_test
import csv



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

        self.gcn_adj_layer2 = GCN(n_h, n_h)
        self.gcn_diff_layer2 = GCN(n_h, n_h)

    def forward(self, seq_pos, seq_neg, adj, diff, sparse, msk, samp_bias1,
                samp_bias2):  # GH: seq1=features ,seq2=shuffled features

        # layer1
        h_adj_layer1_pos = self.gcn_adj_layer1(seq_pos, adj)
        h_diff_layer1_pos = self.gcn_diff_layer1(seq_pos, diff)

        h_adj_layer1_neg = self.gcn_adj_layer1(seq_neg, adj)
        h_diff_layer1_neg = self.gcn_diff_layer1(seq_neg, diff)

        # layer2
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


def load_features(path, format='json'):
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


def train(feature_gnn_path, class_num='binary', prediction='Solution Additives_M', matching_path='graph_match_judge.npy',
          MLF=True):
    # feature_gnn1 = load_features("./graphormer_output_np101-206.npy", format = 'npy')
    # feature_gnn2 = load_features("./graphormer_output_np0-100.npy", format='npy')
    # feature_gnn = np.concatenate([feature_gnn1,feature_gnn2],axis=0)

    feature_gnn = load_features(feature_gnn_path, format='npy')

    feature_gnn = torch.FloatTensor(feature_gnn).cuda()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    feature_exps = np.zeros((334, 10), dtype=float)
    labels = np.zeros((334), dtype=int)
    feature_attribute = load_features("./PC_experiment_json.json", format='json')
    feature_graph_matching = load_features(matching_path, format='npy')
    feature_graph_matching = torch.FloatTensor(feature_graph_matching).cuda()
    counter = 0
    for item in feature_attribute:
        feature_exps[counter][1] = item['Double Metal Core']
        feature_exps[counter][2] = item['Metal Core Type']
        feature_exps[counter][3] = item['Electron Donating Group']
        feature_exps[counter][4] = item['Electron Withdrawing Group']
        feature_exps[counter][5] = item['Proton Donor']

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
        if prediction == "TON":
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
                labels[counter] = 0
            if 50 < sum_TON <= 500:
                labels[counter] = 1
            if sum_TON > 500:
                labels[counter] = 2

        # CAC
        if prediction == "CAC":
            if item['Catalyst Concentration_uM'] == '-' or item['Catalyst Concentration_uM'] == '':
                C = 0.0
            else:
                C = float(item['Catalyst Concentration_uM'])
            if C <= 1:
                labels[counter] = 0
            if C > 1 and C <= 10:
                labels[counter] = 1
            if C > 10 and C <= 100:
                labels[counter] = 2
            if C > 100 and C <= 1000:
                labels[counter] = 3
            if C > 1000:
                labels[counter] = 4

        if prediction == "Photosensitizer":
            labels[counter] = feature_exps[counter][7]

        if prediction == "Solution Additives_M":
            labels[counter] = feature_exps[counter][9]

            # print("label is", labels[counter])
        if prediction == "cuihuaji":
            labels[counter] = item['Evaluation']

        counter = counter + 1

    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    feature_exps = torch.FloatTensor(feature_exps).cuda()
    # idx_train = np.arange(100)
    # idx_test = np.arange(101,334)

    # classes number
    labels = torch.LongTensor(labels).cuda()
    if class_num == 'binary':
        labels = torch.where(labels > 0, 1, 0)
    else:
        pass
    idx_train = np.linspace(0, 332, num=167, endpoint=True, dtype=int)
    idx_test = np.linspace(1, 333, num=167, endpoint=True, dtype=int)
    idx_train = torch.LongTensor(idx_train).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()

    xent = nn.CrossEntropyLoss()

    feature_zeros = np.zeros((334, 8), dtype=float)
    feature_zeros = torch.FloatTensor(feature_zeros).cuda()


    embeds = torch.cat((feature_exps, feature_graph_matching), 1)


    if prediction == "Photosensitizer" or prediction == "Solution Additives_M":
        embeds = torch.cat((feature_exps[:, 0:5], feature_graph_matching), 1)

    # embeds = feature_gnn
    # embeds = feature_exps
    # embeds = feature_zeros

    # embeds = embeds.squeeze()
    log_len = embeds.shape[1]
    train_embs = embeds[idx_train, :]
    test_embs = embeds[idx_test, :]

    #
    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]
    #
    accs = []

    acc_graphs = []

    for count1 in range(50):
        log = LogReg(log_len, 4)  # GH: feature size , class num
        if prediction == "CAC":
            log = LogReg(log_len, 5)
        if prediction == "Photosensitizer":
            log = LogReg(log_len, 15)
        if prediction == "Solution Additives_M":
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
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

        if count1 % 10 == 0: print("acc is: ", acc)

        accs.append(acc)

    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item())

    return accs.mean().item()


if __name__ == "__main__":

    accs = []
    for i in range(5):
        cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
        print(cmd)
        os.system(cmd)
        accs.append(train("./graphormer_output_np_ori_test.npy", class_num='multi', prediction = 'TON'))
    accs = np.array(accs)
    print("=== result === \n MFE test acc TON is ", accs.mean(), " std is ", accs.std())

    accs = []
    for i in range(5):
        cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
        print(cmd)
        os.system(cmd)
        accs.append(train("./graphormer_output_np_ori_test.npy", class_num='multi', prediction = 'CAC'))
    accs = np.array(accs)
    print("=== result === \n MFE test acc CAC is ", accs.mean(), " std is ", accs.std())

    accs = []
    for i in range(5):
        cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
        print(cmd)
        os.system(cmd)
        accs.append(train("./graphormer_output_np_ori_test.npy", class_num='multi', prediction = 'Photosensitizer'))
    accs = np.array(accs)
    print("=== result === \n MFE test acc Photosensitizer is ", accs.mean(), " std is ", accs.std())

    accs = []
    for i in range(5):
        cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
        print(cmd)
        os.system(cmd)
        accs.append(train("./graphormer_output_np_ori_test.npy", class_num='multi', prediction = 'Solution Additives_M'))
    accs = np.array(accs)
    print("=== result === \n MFE test acc Solution Additives_M is ", accs.mean(), " std is ", accs.std())



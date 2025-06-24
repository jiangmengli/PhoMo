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


def train(graph_search_path):

    # feature_gnn1 = load_features("./graphormer_output_np101-206.npy", format = 'npy')
    # feature_gnn2 = load_features("./graphormer_output_np0-100.npy", format='npy')
    # feature_gnn = np.concatenate([feature_gnn1,feature_gnn2],axis=0)

    # feature_gnn = load_features(feature_gnn_path, format='npy')
    # feature_gnn_drop = load_features(feature_gnn_drop_path, format='npy')
    #
    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    # feature_gnn_drop = torch.FloatTensor(feature_gnn_drop).cuda()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    feature_exps = np.zeros((334,8),dtype=float)
    labels = np.zeros((334),dtype=int)
    feature_attribute = load_features("./graphsearch/drop_109"
                    "/experiment_oriented/PC_experiment_json.json", format = 'json')
    counter = 0
    for item in feature_attribute:
        feature_exps[counter][1] = item['Double Metal Core']
        feature_exps[counter][2] = item['Metal Core Type']
        feature_exps[counter][3] = item['Electron Donating Group']
        feature_exps[counter][4] = item['Electron Withdrawing Group']
        feature_exps[counter][5] = item['Proton Donor']
        labels[counter] = item['Evaluation']
        counter = counter + 1

    feature_exps_drop = np.zeros((334,8),dtype=float)
    feature_attribute_drop = load_features(graph_search_path, format = 'json')
    counter = 0
    for item in feature_attribute_drop:
        feature_exps_drop[counter][1] = item['Double Metal Core']
        feature_exps_drop[counter][2] = item['Metal Core Type']
        feature_exps_drop[counter][3] = item['Electron Donating Group']
        feature_exps_drop[counter][4] = item['Electron Withdrawing Group']
        feature_exps_drop[counter][5] = item['Proton Donor']
        counter = counter + 1


    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    # feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    # feature_exps = torch.FloatTensor(feature_exps).cuda()
    # idx_train = np.arange(100)
    # idx_test = np.arange(101,334)

    feature_exps = torch.FloatTensor(feature_exps).cuda()
    feature_exps_drop = torch.FloatTensor(feature_exps_drop).cuda()





    # classes number
    labels = torch.LongTensor(labels).cuda()
    labels = torch.where(labels>0, 1,0)
    idx_train = np.linspace(0, 332, num=167, endpoint=True, dtype=int)
    idx_test = np.linspace(1, 333, num=167, endpoint=True, dtype=int)
    idx_train = torch.LongTensor(idx_train).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()

    xent = nn.CrossEntropyLoss()

    feature_zeros = np.zeros((334, 8), dtype=float)
    feature_zeros = torch.FloatTensor(feature_zeros).cuda()

    # embeds = torch.cat((feature_exps, feature_gnn), 1)
    embeds = feature_exps
    # embeds = feature_exps
    # embeds = feature_zeros

    # embeds = embeds.squeeze()
    log_len = embeds.shape[1]
    train_embs = embeds[idx_train, :]
    test_embs = embeds[idx_test, :]

    embeds_drop = feature_exps_drop
    test_embs_drop = embeds_drop[idx_test, :]

    feature_diff = cos(embeds, embeds_drop)

    #
    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]
    #
    accs = []
    accs_drop = []
    acc_graphs = []
    acc_graphs_drop = []

    for count1 in range(50):
        log = LogReg(log_len , 4) # GH: feature size , class num
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
        acc_graph = (preds == test_lbls).float()

        logits_drop = log(test_embs_drop)
        preds_drop = torch.argmax(logits_drop, dim=1)
        acc_drop = torch.sum(preds_drop == test_lbls).float() / test_lbls.shape[0]
        acc_graph_drop = (preds_drop == test_lbls).float()

        accs.append(acc * 100)
        acc_graphs.append(acc_graph)
        accs_drop.append(acc_drop * 100)
        acc_graphs_drop.append(acc_graph_drop)

        if count1 % 10 == 0:print("acc is: ", acc, " acc_drop is: ", acc_drop)

    accs = torch.stack(accs)
    acc_graphs = torch.stack(acc_graphs)
    cal_times = acc_graphs.shape[0]
    acc_graphs = acc_graphs.sum(dim=0)/cal_times
    print(accs.mean().item(), accs.std().item())

    accs_drop = torch.stack(accs_drop)
    acc_graphs_drop = torch.stack(acc_graphs_drop)
    acc_graphs_drop = acc_graphs_drop.sum(dim=0) / cal_times
    print(accs_drop.mean().item(), accs_drop.std().item())

    acc_diff_0 = acc_graphs_drop - acc_graphs


    idx_train = np.linspace(1, 333, num=167, endpoint=True, dtype=int)
    idx_test = np.linspace(0, 332, num=167, endpoint=True, dtype=int)
    idx_train = torch.LongTensor(idx_train).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()



    xent = nn.CrossEntropyLoss()

    feature_zeros = np.zeros((334, 8), dtype=float)
    feature_zeros = torch.FloatTensor(feature_zeros).cuda()

    # embeds = torch.cat((feature_exps, feature_gnn), 1)
    embeds = feature_exps
    # embeds = feature_exps
    # embeds = feature_zeros

    # embeds = embeds.squeeze()
    log_len = embeds.shape[1]
    train_embs = embeds[idx_train, :]
    test_embs = embeds[idx_test, :]

    embeds_drop = feature_exps_drop
    test_embs_drop = embeds_drop[idx_test, :]
    #
    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    accs_drop = []
    acc_graphs = []
    acc_graphs_drop = []

    for count1 in range(50):
        log = LogReg(log_len , 4) # GH: feature size , class num
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
        acc_graph = (preds == test_lbls).float()

        logits_drop = log(test_embs_drop)
        preds_drop = torch.argmax(logits_drop, dim=1)
        acc_drop = torch.sum(preds_drop == test_lbls).float() / test_lbls.shape[0]
        acc_graph_drop = (preds_drop == test_lbls).float()

        accs.append(acc * 100)
        acc_graphs.append(acc_graph)
        accs_drop.append(acc_drop * 100)
        acc_graphs_drop.append(acc_graph_drop)

        if count1 % 10 == 0:print("acc is: ", acc, " acc_drop is: ", acc_drop)

    accs = torch.stack(accs)
    acc_graphs = torch.stack(acc_graphs)
    cal_times = acc_graphs.shape[0]
    acc_graphs = acc_graphs.sum(dim=0)/cal_times
    print(accs.mean().item(), accs.std().item())

    accs_drop = torch.stack(accs_drop)
    acc_graphs_drop = torch.stack(acc_graphs_drop)
    acc_graphs_drop = acc_graphs_drop.sum(dim=0) / cal_times
    print(accs_drop.mean().item(), accs_drop.std().item())

    acc_diff_1 = acc_graphs_drop - acc_graphs

    acc_diff_all = torch.zeros(334).cuda()
    acc_diff_all[idx_test] = acc_diff_1
    acc_diff_all[idx_train] = acc_diff_0







    # accs.append(acc * 100)
    # accs = torch.stack(accs)
    # print(accs.mean().item(), accs.std().item())


    return feature_diff, acc_diff_all


# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore")
#     torch.cuda.set_device(0)
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='Disables CUDA training.')
#     parser.add_argument('--fastmode', action='store_true', default=False,
#                         help='Validate during training pass.')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed.')
#     parser.add_argument('--epochs', type=int, default=200,
#                         help='Number of epochs to train.')
#     parser.add_argument('--lr', type=float, default=0.01,
#                         help='Initial learning rate.')
#     parser.add_argument('--weight_decay', type=float, default=5e-4,
#                         help='Weight decay (L2 loss on parameters).')
#     parser.add_argument('--hidden', type=int, default=16,
#                         help='Number of hidden units.')
#     parser.add_argument('--dropout', type=float, default=0.5,
#                         help='Dropout rate (1 - keep probability).')
#     parser.add_argument('--feature_gnn_path', default='./graphormer_output_np0-100.npy',
#                         help='feature_gnn_path).')
#     parser.add_argument('--feature_attribute_path', default='./experiment_oriented/PC_experiment_json.json',
#                         help='feature_attribute_path).')
#
#     args = parser.parse_args()
#
#     acc_sum = 0
#     acc_max = 0
#     acc_min = 100
#     for i in range(50):
#         acc = train(args)
#         acc_sum = acc_sum + acc
#         if acc_max < acc:
#             acc_max = acc
#         if acc_min > acc:
#             acc_min = acc
#         print("GH: round " , i , "acc average is: " , acc_sum/(i+1))
#         print("GH: round ", i, "acc max is: ", acc_max)
#         print("GH: round ", i, "acc min is: ", acc_min)


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

import both_test

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

def train(feature_path, type="gnn"):
    feature_gnn = load_features(feature_path, format = 'npy')

    feature_exps = np.zeros((207,8),dtype=float)
    labels = np.zeros((207),dtype=int)
    feature_attribute = load_features("./dataset/experiment_oriented/PC_experiment_json.json", format = 'json')
    counter = 0
    for item in feature_attribute:
        feature_exps[counter][1] = item['Double Metal Core']
        feature_exps[counter][2] = item['Metal Core Type']
        feature_exps[counter][3] = item['Electron Donating Group']
        feature_exps[counter][4] = item['Electron Withdrawing Group']
        feature_exps[counter][5] = item['Proton Donor']
        labels[counter] = item['Evaluation']
        counter = counter + 1

    feature_gnn = torch.FloatTensor(feature_gnn).cuda()
    feature_exps = torch.FloatTensor(feature_exps).cuda()

    # classes number
    labels = torch.LongTensor(labels).cuda()
    labels = torch.where(labels>0, 1,0)
    idx_train = np.linspace(0, 205, num=103, endpoint=True, dtype=int)
    idx_test = np.linspace(1, 206, num=103, endpoint=True, dtype=int)


    idx_train = torch.LongTensor(idx_train).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()

    xent = nn.CrossEntropyLoss()

    feature_zeros = np.zeros((207, 8), dtype=float)
    feature_zeros = torch.FloatTensor(feature_zeros).cuda()

    if type == 'gnn':
        embeds = feature_gnn
    if type == 'graph_search':
        embeds = feature_exps
    if type == 'all':
        embeds = torch.cat((feature_exps, feature_gnn), 1)

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
    # wd = 0.01 if dataset == 'citese.bu' else 0.0
    #
    #
    #
    for count1 in range(50):
        log = LogReg(log_len , 4) # GH: feature size , class num
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
        log.cuda()
        for count2 in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        if count1 % 10 == 0:print("train: ", acc)

    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item())
    return accs.mean().item()


if __name__ == '__main__':
    print("======= process start! =======\n")


    cmd = "rm ./dataset/cuihuaji/raw/*"
    os.system(cmd)

    cmd = "rm ./dataset/cuihuaji/processed/*"
    os.system(cmd)

    cmd = "rm ./PC_experiment_json.json"
    os.system(cmd)

    cmd = "cp ./data_safecase/cuihuaji/* ./dataset/cuihuaji/raw/"
    os.system(cmd)

    cmd = "cp ./data_safecase/cuihuaji/PC_experiment_json.json ./PC_experiment_json.json"
    os.system(cmd)

    print("======= step 1: calculate gnn scores =======\n")

    skip = False

    if not skip:

        os.environ['MKL_THREADING_LAYER'] = 'GNU'

        accs = []
        for i in range(0):
            cmd = "rm ./Graphormer-main/graphormer/graphormer_output_np_drop1000.npy"
            print(cmd)
            os.system(cmd)

            cmd = "cd ./Graphormer-main/graphormer && /home/iscas/miniconda3/envs/gcl/bin/python ./entry.py --num_workers 8 --seed 1 --batch_size 334 --dataset_name cuihuaji --accelerator cpu --precision 32 --ffn_dim 768 " + \
                " --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.3 --dropout_rate 0.1 --weight_decay 0.01 " + \
                "--n_layers 12 --multi_hop_max_dist 5 --default_root_dir ./Graphormer-main/graphormer --tot_updates 1000" + \
                " --warmup_updates 3417 --max_epochs 1 --peak_lr 2e-4 " + \
                "--checkpoint_path /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/Graphormer-main/model/PCQM4M-LSC-epoch=424-valid_mae=0.1277.ckpt " + \
                "--edge_type multi_hop --end_lr 1e-9 --flag --flag_m 4 --flag_step_size 0.001 --flag_mag 0.001 " + \
                "--end_lr 2"
            os.system(cmd)
            cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
            print(cmd)
            os.system(cmd)
            accs.append(feature_test.train("./graphormer_output_np_ori_test.npy", class_num='multi'))
        accs = np.array(accs)
        print("=== result === \n graphormer multi class feature test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(0):
            cmd = "rm ./Graphormer-main/graphormer/graphormer_output_np_drop1000.npy"
            print(cmd)
            os.system(cmd)

            cmd = "cd ./Graphormer-main/graphormer && /home/iscas/miniconda3/envs/gcl/bin/python ./entry.py --num_workers 8 --seed 1 --batch_size 334 --dataset_name cuihuaji --accelerator cpu --precision 32 --ffn_dim 768 " + \
                " --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.3 --dropout_rate 0.1 --weight_decay 0.01 " + \
                "--n_layers 12 --multi_hop_max_dist 5 --default_root_dir ./Graphormer-main/graphormer --tot_updates 1000" + \
                " --warmup_updates 3417 --max_epochs 1 --peak_lr 2e-4 " + \
                "--checkpoint_path /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/Graphormer-main/model/PCQM4M-LSC-epoch=424-valid_mae=0.1277.ckpt " + \
                "--edge_type multi_hop --end_lr 1e-9 --flag --flag_m 4 --flag_step_size 0.001 --flag_mag 0.001 " + \
                "--end_lr 2"
            os.system(cmd)
            cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
            print(cmd)
            os.system(cmd)
            accs.append(feature_test.train("./graphormer_output_np_ori_test.npy", class_num='binary'))
        accs = np.array(accs)
        print("=== result === \n graphormer binary class feature test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(0):
            accs.append(graph_search_test.train("./graphormer_output_np_ori_test.npy", class_num='multi'))
        accs = np.array(accs)
        print("=== result === \n graphsearch multi class test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(0):
            accs.append(graph_search_test.train("./graphormer_output_np_ori_test.npy", class_num='binary'))
        accs = np.array(accs)
        print("=== result === \n graphsearch binary class test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):
            cmd = "rm ./Graphormer-main/graphormer/graphormer_output_np_drop1000.npy"
            print(cmd)
            os.system(cmd)

            cmd = "cd ./Graphormer-main/graphormer && /home/iscas/miniconda3/envs/gcl/bin/python ./entry.py --num_workers 8 --seed 1 --batch_size 334 --dataset_name cuihuaji --accelerator cpu --precision 32 --ffn_dim 768 " + \
                " --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.3 --dropout_rate 0.1 --weight_decay 0.01 " + \
                "--n_layers 12 --multi_hop_max_dist 5 --default_root_dir ./Graphormer-main/graphormer --tot_updates 1000" + \
                " --warmup_updates 3417 --max_epochs 1 --peak_lr 2e-4 " + \
                "--checkpoint_path /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/Graphormer-main/model/PCQM4M-LSC-epoch=424-valid_mae=0.1277.ckpt " + \
                "--edge_type multi_hop --end_lr 1e-9 --flag --flag_m 4 --flag_step_size 0.001 --flag_mag 0.001 " + \
                "--end_lr 2"
            os.system(cmd)
            cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
            print(cmd)
            os.system(cmd)
            accs.append(both_test.train("./graphormer_output_np_ori_test.npy", class_num='multi'))
        accs = np.array(accs)
        print("=== result === \n graphormer multi class both test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):
            cmd = "rm ./Graphormer-main/graphormer/graphormer_output_np_drop1000.npy"
            print(cmd)
            os.system(cmd)

            cmd = "cd ./Graphormer-main/graphormer && /home/iscas/miniconda3/envs/gcl/bin/python ./entry.py --num_workers 8 --seed 1 --batch_size 334 --dataset_name cuihuaji --accelerator cpu --precision 32 --ffn_dim 768 " + \
                " --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.3 --dropout_rate 0.1 --weight_decay 0.01 " + \
                "--n_layers 12 --multi_hop_max_dist 5 --default_root_dir ./Graphormer-main/graphormer --tot_updates 1000" + \
                " --warmup_updates 3417 --max_epochs 1 --peak_lr 2e-4 " + \
                "--checkpoint_path /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/Graphormer-main/model/PCQM4M-LSC-epoch=424-valid_mae=0.1277.ckpt " + \
                "--edge_type multi_hop --end_lr 1e-9 --flag --flag_m 4 --flag_step_size 0.001 --flag_mag 0.001 " + \
                "--end_lr 2"
            os.system(cmd)
            cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_ori_test.npy"
            print(cmd)
            os.system(cmd)
            accs.append(both_test.train("./graphormer_output_np_ori_test.npy", class_num='binary'))
        accs = np.array(accs)
        print("=== result === \n graphormer binary class both test acc is ", accs.mean(), " std is ", accs.std())



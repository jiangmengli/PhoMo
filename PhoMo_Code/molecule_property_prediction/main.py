import data_generate.create_csv_node_drop as create_data
import os
import numpy as np
import feature_get
import graph_search_get
import torch
import csv
import both_get

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


if __name__ == '__main__':

    cur_num=90

    print("======= process start! =======\n")

    print("======= step 1: generating data =======\n")

    skip=False

    if not skip:
        create_data.create_csvs()

    print("======= step 2: get output features of graphormer =======\n")

    skip=False

    if not skip:

        # for i in range(110):
        for i in range(cur_num):

            print("\n\n============ edge "+str(i)+" ===============\n\n")

            os.environ['MKL_THREADING_LAYER'] = 'GNU'

            cmd = "rm -r ./dataset/cuihuaji/raw/"
            print(cmd)
            os.system(cmd)

            cmd = "rm -r ./dataset/cuihuaji/processed/ && rm ./Graphormer-main/graphormer/graphormer_output_np.npy"
            print(cmd)
            os.system(cmd)

            cmd = "mkdir ./dataset/cuihuaji/raw/"
            print(cmd)
            os.system(cmd)

            cmd = "cp ./data_generate/data_for_graphormer/raw/drop_"+str(i)+"/* ./dataset/cuihuaji/raw"
            print(cmd)
            os.system(cmd)

            cmd = "gzip ./dataset/cuihuaji/raw/*"
            print(cmd)
            print(os.getcwd())
            os.system(cmd)

            cmd = "cd ./Graphormer-main/graphormer && /home/iscas/miniconda3/envs/gcl/bin/python ./entry.py --num_workers 8 --seed 1 --batch_size 334 --dataset_name cuihuaji --accelerator cpu --precision 32 --ffn_dim 768 " + \
                " --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.3 --dropout_rate 0.1 --weight_decay 0.01 " + \
                "--n_layers 12 --multi_hop_max_dist 5 --default_root_dir ./Graphormer-main/graphormer --tot_updates "+ str(i) + \
                " --warmup_updates 3417 --max_epochs 1 --peak_lr 2e-4 " + \
                "--checkpoint_path /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/Graphormer-main/model/PCQM4M-LSC-epoch=424-valid_mae=0.1277.ckpt " + \
                "--edge_type multi_hop --end_lr 1e-9 --flag --flag_m 4 --flag_step_size 0.001 --flag_mag 0.001 " + \
                "--end_lr 2"

            os.system(cmd)

            cmd = "cp ./Graphormer-main/graphormer/graphormer_output_np.npy ./graphormer_output_np_save_drop"+str(i)+".npy"
            print(cmd)
            os.system(cmd)

    print("======= step 3: compare output features of graphormer =======\n")

    skip=False

    if not skip:


        feature_diffs = []
        acc_diffs = []

        for i in range(cur_num):
            feature_diff, acc_diff =\
                feature_get.train("graphormer_output_np_save_drop109.npy", "graphormer_output_np_save_drop"+str(i)+".npy")
            print(" ")

            feature_diffs.append(feature_diff)
            acc_diffs.append(acc_diff)

        feature_diffs = torch.stack(feature_diffs)
        acc_diffs = torch.stack(acc_diffs)

        feature_diffs = torch.transpose(feature_diffs, 0, 1)
        feature_diffs = 1-feature_diffs
        acc_diffs = torch.transpose(acc_diffs, 0, 1)

        feature_diffs_np = feature_diffs.detach().cpu().numpy()
        np.save(f'./GNN_feature_diffs.npy', feature_diffs_np)
        np.savetxt("./GNN_feature_diffs.csv",feature_diffs_np,delimiter=',')

        acc_diffs_np = acc_diffs.detach().cpu().numpy()
        np.save(f'./GNN_acc_diffs.npy', acc_diffs_np)
        np.savetxt("./GNN_acc_diffs.csv",acc_diffs_np,delimiter=',')


    print("======= step 4: compare output of graphsearch =======\n")

    skip=False

    if not skip:

        feature_diffs = []
        acc_diffs = []

        for i in range(cur_num):
            feature_diff, acc_diff =\
                graph_search_get.train("./graph_search_node_drop/drop_"+str(i)+
                                       "/experiment_oriented/PC_experiment_json.json")
            print(" ")

            feature_diffs.append(feature_diff)
            acc_diffs.append(acc_diff)

            feature_diff_np = feature_diff.detach().cpu().numpy()
            np.save(f'./graph_search_node_drop/feature_diff'+str(i)+'.npy', feature_diff_np)
            np.savetxt("./graph_search_node_drop/feature_diff"+str(i)+".csv", feature_diff_np, delimiter=',')

            acc_diff_np = acc_diff.detach().cpu().numpy()
            np.save(f'./graph_search_node_drop/acc_diff'+str(i)+'.npy', acc_diff_np)
            np.savetxt("./graph_search_node_drop/acc_diff"+str(i)+".csv", acc_diff_np, delimiter=',')

        feature_diffs = torch.stack(feature_diffs)
        acc_diffs = torch.stack(acc_diffs)

        feature_diffs = torch.transpose(feature_diffs, 0, 1)
        feature_diffs = 1-feature_diffs
        acc_diffs = torch.transpose(acc_diffs, 0, 1)

        feature_diffs_np = feature_diffs.detach().cpu().numpy()
        np.save(f'./graph_search_feature_diffs.npy', feature_diffs_np)
        np.savetxt("./graph_search_feature_diffs.csv",feature_diffs_np,delimiter=',')

        acc_diffs_np = acc_diffs.detach().cpu().numpy()
        np.save(f'./graph_search_acc_diffs.npy', acc_diffs_np)
        np.savetxt("./graph_search_acc_diffs.csv",acc_diffs_np,delimiter=',')

    print("======= step 5: compare output of graphsearch+GNN =======\n")

    skip=False

    if not skip:

        feature_diffs = []
        acc_diffs = []

        for i in range(cur_num):
            feature_diff, acc_diff = \
                both_get.train("graphormer_output_np_save_drop109.npy",
                               "graphormer_output_np_save_drop"+str(i)+".npy",
                               "./graph_search_node_drop/drop_" + str(i) +
                                       "/experiment_oriented/PC_experiment_json.json")
            print(" ")

            feature_diffs.append(feature_diff)
            acc_diffs.append(acc_diff)

            feature_diff_np = feature_diff.detach().cpu().numpy()
            np.save(f'./both/both_feature_diff' + str(i) + '.npy', feature_diff_np)
            np.savetxt("./both/both_feature_diff" + str(i) + ".csv", feature_diff_np, delimiter=',')

            acc_diff_np = acc_diff.detach().cpu().numpy()
            np.save(f'./both/both_acc_diff' + str(i) + '.npy', acc_diff_np)
            np.savetxt("./both/both_acc_diff" + str(i) + ".csv", acc_diff_np, delimiter=',')

        feature_diffs = torch.stack(feature_diffs)
        acc_diffs = torch.stack(acc_diffs)

        feature_diffs = torch.transpose(feature_diffs, 0, 1)
        feature_diffs = 1 - feature_diffs
        acc_diffs = torch.transpose(acc_diffs, 0, 1)

        feature_diffs_np = feature_diffs.detach().cpu().numpy()
        np.save(f'./both_feature_diffs.npy', feature_diffs_np)
        np.savetxt("./both_feature_diffs.csv", feature_diffs_np, delimiter=',')

        acc_diffs_np = acc_diffs.detach().cpu().numpy()
        np.save(f'./both_acc_diffs.npy', acc_diffs_np)
        np.savetxt("./both_acc_diffs.csv", acc_diffs_np, delimiter=',')

    # print("======= step 6: data process =======\n")
    #
    # feature_acc_diffs = load_features("GNN_acc_diffs.npy", format='npy')
    #
    # with open("data_generate/data_for_graphormer/raw/original_csvs/num-edge-list.csv") as f:
    #     reader = csv.reader(f)
    #     num_edge_list = [row for row in reader]
    #
    # for i in range(334):
    #     print("edge num is ", num_edge_list[i][0])
    #     n = num_edge_list[i][0]
    #     feature_acc_diffs[i]
    #
    # print("===")


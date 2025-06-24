import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from tqdm import tqdm
import torch
import train


class CHJ_Dataset(object):
    def __init__(self, root='/home/gaohang/Researches/ChemGraph/GIN-PCQimp/dataset', smiles2graph=smiles2graph):
        print('The PCQM4M has been deprecated. The leaderboard is no longer maintained.')
        print('Please use PCQM4Mv2 instead.')

        '''
        DGL PCQM4M dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'cuihuaji')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 5144ebaa7c67d24da1a2acbe41f57f6a
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m_kddcup2021.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-acceazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'

        # check version and update if necessary
        # if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
        #     print('PCQM4M dataset has been updated.')
        #     if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
        #         shutil.rmtree(self.folder)

        super(CHJ_Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        self.prepare_graph()

    # def download(self):
    #     if decide_download(self.url):
    #         path = download_url(self.url, self.original_root)
    #         extract_zip(path, self.original_root)
    #         os.unlink(path)
    #     else:
    #         print('Stop download.')
    #         exit(-1)

    def prepare_graph(self):
        print("start graph matching!")
        # read graphs
        # processed_dir = osp.join(self.folder, 'processed')
        raw_dir = './matching'
        # pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')
        # graphs_, label_dict_ = torch.load(
        #     '/home/gaohang/Researches/ChemGraph/dataset/cuihuaji/processed/geometric_data_processed.pt')

        node_features = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), header=None).values
        edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), header=None).values
        edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), header=None).values
        graph_label = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), header=None).values
        molecul_idx = graph_label[:,0]
        num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), header=None)
        num_node_list = np.array(num_node_list)
        num_node_list = num_node_list.tolist()
        num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), header=None)
        num_edge_list = np.array(num_edge_list)
        num_edge_list = num_edge_list.tolist()

        # read substructures:
        # read graphs
        raw_dir = "./first_substructure/first_substructure/"
        subs_node_features = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), header=None).values
        subs_edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), header=None).values
        subs_edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), header=None).values
        subs_num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), header=None)
        subs_num_node_list = np.array(subs_num_node_list)
        subs_num_node_list = subs_num_node_list.tolist()
        subs_num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), header=None)
        subs_num_edge_list = np.array(subs_num_edge_list)
        subs_num_edge_list = subs_num_edge_list.tolist()

        edge_count = 0
        node_count = 0

        judge = []

        counter = 0

        print("matching ", len(num_edge_list), " graphs")

        for edge_num, node_num in zip(num_edge_list, num_node_list):

            edge_num=edge_num[0]
            node_num=node_num[0]

            if counter > 0:
                if molecul_idx[counter] == molecul_idx[counter-1]:
                    print(judge[counter-1])
                    judge.append(judge[counter-1])
                    counter = counter + 1
                    edge_count = edge_count + edge_num
                    node_count = node_count + node_num
                    continue


            counter = counter + 1

            print("\n ===== match graph " , counter , " ====== \n")



            g_edge = edge[[x for x in range(edge_count, edge_count + edge_num)]]
            g_edge_feat = edge_feat[[x for x in range(edge_count, edge_count + edge_num)]]
            g_node_features = node_features[[x for x in range(node_count, node_count + node_num)]]

            g_node_features = torch.from_numpy(g_node_features).cuda()
            g_edge = torch.from_numpy(g_edge).cuda()
            g_edge = g_edge.transpose(0, 1)
            g_edge_feat = torch.from_numpy(g_edge_feat).cuda()

            score_list = []

            edge_count_sub = 0
            node_count_sub = 0

            for edge_num_sub, node_num_sub in zip(subs_num_edge_list, subs_num_node_list):

                # print(" === sub === ")

                edge_num_sub = edge_num_sub[0]
                node_num_sub = node_num_sub[0]
                g_subs_edge = subs_edge[[x for x in range(edge_count_sub, (edge_count_sub + edge_num_sub))],:]
                g_subs_edge_feat = subs_edge_feat[[x for x in range(edge_count_sub, edge_count_sub + edge_num_sub)]]
                g_subs_node_features = subs_node_features[[x for x in range(node_count_sub, node_count_sub + node_num_sub)]]

                g_subs_node_features = torch.from_numpy(g_subs_node_features).cuda()
                g_subs_edge = torch.from_numpy(g_subs_edge).cuda()
                g_subs_edge = g_subs_edge.transpose(0, 1)
                g_subs_edge_feat = torch.from_numpy(g_subs_edge_feat).cuda()

                # print("input of score is ", g_node_features, g_edge, g_subs_node_features, g_subs_edge,
                #             g_edge_feat, g_subs_edge_feat)
                score = train.train(g_node_features, g_edge, g_subs_node_features, g_subs_edge,
                            g_edge_feat, g_subs_edge_feat)

                # print("score is ", score)

                score_list.append(score)

                edge_count_sub = edge_count_sub + edge_num_sub
                node_count_sub = node_count_sub + node_num_sub

            idx = 0
            max = 0

            score_list = np.round(score_list, 2)
            # print("score_list is", score_list )

            for x in range(9):
                if max < score_list[x]:
                    max = score_list[x]
                    idx = x

            judge.append([idx + 1,max])

            edge_count = edge_count + edge_num
            node_count = node_count + node_num

            # print(" judge is ", judge[0])


        np.save(f'./graph_match_judge.npy', judge)




if __name__ == '__main__':
    CHJ_Dataset()

    # split_dict = dataset.get_idx_split()
    # print(split_dict)
    # print(dataset[split_dict['train']])
    # print(collate_dgl([dataset[0], dataset[1], dataset[2]]))

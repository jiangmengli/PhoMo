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


class DglPCQM4MDataset(object):
    def __init__(self, smiles2graph=smiles2graph):
        print('The PCQM4M has been deprecated. The leaderboard is no longer maintained.')
        print('Please use PCQM4Mv2 instead.')

        '''
        DGL PCQM4M dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        '''

        self.smiles2graph = smiles2graph
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

        super(DglPCQM4MDataset, self).__init__()

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
        processed_dir = osp.join('./dataset/cuihuaji/', 'processed')
        os.system('rm '+processed_dir+'/*')
        raw_dir = './dataset/cuihuaji/raw'
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
            # if pre-processed file already exis6
            # ts
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']
        else:
            # if pre-processed file does not exist

            # if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
            #     # if the raw file does not exist, then download it.
            #     self.download()

            # data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            node_features = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), header=None)
            edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), header=None)
            edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), header=None)
            graph_label = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), header=None)
            num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), header=None)
            num_node_list = np.array(num_node_list)
            num_node_list = num_node_list.tolist()
            num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), header=None)
            num_edge_list = np.array(num_edge_list)
            num_edge_list = num_edge_list.tolist()
            # smiles_list = data_df['smiles']
            # homolumogap_list = data_df['homolumogap']
            # num_node_list = num_node_list.loc[:, "a"]
            # num_edge_list = num_edge_list.loc[:, "a"]

            print('generate graphs...')
            self.graphs = []
            self.labels = []
            num_node_count = 0
            num_edge_count = 0
            graph_count = 1
            for node_i, edge_i in zip(num_node_list, num_edge_list):
                print("graph count is:", graph_count)
                # if graph_count == 28:
                #     print(" ")
                graph_count = graph_count + 1
                node_i = int(node_i[0])
                edge_i = int(edge_i[0])
                num_node_count_start = num_node_count
                num_edge_count_start = num_edge_count
                num_node_count_stop = num_node_count + node_i - 1
                num_edge_count_stop = num_edge_count + edge_i - 1

                num_node_count = num_node_count + node_i
                num_edge_count = num_edge_count + edge_i

                edge_index_df = edge.loc[num_edge_count_start:num_edge_count_stop]
                edge_index_graph = edge_index_df.values
                edge_index_graph = np.swapaxes(edge_index_graph, 0, 1)

                edge_feat_df = edge_feat.loc[num_edge_count_start:num_edge_count_stop]
                edge_feat_graph = edge_feat_df.values

                node_feat_df = node_features.loc[num_node_count_start:num_node_count_stop]
                node_feat_graph = node_feat_df.values

                # num_nodes = node_i

                graph = {
                    'edge_index':edge_index_graph,
                    'edge_feat':edge_feat_graph,
                    'node_feat':node_feat_graph,
                    'num_nodes':node_i
                }

                # smiles = 0
                homolumogap = 0
                # graph = self.smiles2graph(smiles)
                # graph =

                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])

                # if graph_count == 7:
                #     print(" ")

                dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
                dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)

                self.graphs.append(dgl_graph)
                self.labels.append(homolumogap)

            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # # double-check prediction target
            # split_dict = self.get_idx_split()
            # assert (all([not torch.isnan(self.labels[i]) for i in split_dict['train']]))
            # assert (all([not torch.isnan(self.labels[i]) for i in split_dict['valid']]))
            # assert (all([torch.isnan(self.labels[i]) for i in split_dict['test']]))

            print('Saving...')
            save_graphs(pre_processed_file_path, self.graphs, labels={'labels': self.labels})

            try:
                os.remove('./dataset/pcqm4m_kddcup2021/processed/dgl_data_processed')
            except:
                pass

            save_graphs('./dataset/pcqm4m_kddcup2021/processed/dgl_data_processed', self.graphs, labels={'labels': self.labels})

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.folder, 'split_dict.pt')))
        return split_dict

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


# Collate function for ordinary graph classification
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels


if __name__ == '__main__':
    dataset = DglPCQM4MDataset()

    print(dataset)
    # split_dict = dataset.get_idx_split()
    # print(split_dict)
    # print(dataset[split_dict['train']])
    # print(collate_dgl([dataset[0], dataset[1], dataset[2]]))

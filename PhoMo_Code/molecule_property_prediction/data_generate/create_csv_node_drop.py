# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import pandas as pd






def create_csvs():
    for i in range(90):

        print("start create drop_", i)

        with open("data_safecase/cuihuaji/csv/edge.csv") as f:
            reader = csv.reader(f)
            edge = [row for row in reader]
            # print(rows)

        with open("data_safecase/cuihuaji/csv/edge-feat.csv") as f:
            reader = csv.reader(f)
            edge_feat = [row for row in reader]

        with open("data_safecase/cuihuaji/csv/graph-label.csv") as f:
            reader = csv.reader(f)
            graph_label = [row for row in reader]

        with open("data_safecase/cuihuaji/csv/node-feat.csv") as f:
            reader = csv.reader(f)
            node_feat = [row for row in reader]

        with open("data_safecase/cuihuaji/csv/num-edge-list.csv") as f:
            reader = csv.reader(f)
            num_edge_list = [row for row in reader]

        with open("data_safecase/cuihuaji/csv/num-node-list.csv") as f:
            reader = csv.reader(f)
            num_node_list = [row for row in reader]

        print("files loaded")

        if(i==11):
            print(" ")

        edge_counter = 0
        graph_counter = 0
        for edge_num in num_edge_list:

            if int(edge_num[0]) > i:
                edge_delete = edge_counter + i
                edge.pop(edge_delete)
                edge_feat.pop(edge_delete)
                edge_counter = edge_counter + int(edge_num[0]) - 1
                num_edge_list[graph_counter] = str(int(edge_num[0]) - 1)
            else:
                edge_counter = edge_counter + int(edge_num[0])
                num_edge_list[graph_counter] = str(int(edge_num[0]))

            graph_counter = graph_counter + 1

        print("creating csv.gzs")

        folder_path = 'data_generate/data_for_graphormer/raw/drop_' + str(i) + '/'
        try:
            os.system("rm -rf "+ folder_path)
            os.mkdir(folder_path)
        except:
            print("create dir error!")

        new_edge = []
        new_edge_feat = []
        new_node_feat = []
        new_num_edge_list = []
        new_num_node_list = []
        node_count = 0
        edge_count = 0

        for node_num, edge_num in zip(num_node_list, num_edge_list):
            node_num = int(node_num[0])
            edge_num = int(edge_num)
            node_feat_graph = node_feat[node_count:(node_count + node_num)]
            edge_graph = edge[edge_count:(edge_count + edge_num)]
            edge_feat_graph = edge_feat[edge_count:(edge_count + edge_num)]

            edge_del_list = []

            if i < node_num:
                del node_feat_graph[i]
                counter = 0
                for edge_s in edge_graph:
                    if str(i) in edge_s:
                        edge_del_list.append(counter)
                    counter = counter + 1

                for idx in reversed(edge_del_list):
                    del edge_graph[idx]
                    del edge_feat_graph[idx]

            node_count = node_count + node_num
            edge_count = edge_count + edge_num

            new_node_feat.extend(node_feat_graph)
            new_edge.extend(edge_graph)
            new_edge_feat.extend(edge_feat_graph)
            new_num_node_list.append(len(node_feat_graph))
            new_num_edge_list.append(len(edge_graph))

        # 下面这行代码运行报错
        # name = ['one', 'two', 'three']
        test = pd.DataFrame(data=new_edge)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/edge.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=new_edge_feat)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/edge-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=graph_label)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/graph-label.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=new_node_feat)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/node-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=new_num_edge_list)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/num-edge-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=new_num_node_list)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/num-node-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        


        print("finish create drop_", i,"\n")


if __name__ == '__main__':
    create_csvs()



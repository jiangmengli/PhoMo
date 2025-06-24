# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import pandas as pd






def create_csvs():
    for i in range(110):

        print("start create drop_", i)

        with open("data_generate/data_for_graphormer/raw/original_csvs/edge.csv") as f:
            reader = csv.reader(f)
            edge = [row for row in reader]
            # print(rows)

        with open("data_generate/data_for_graphormer/raw/original_csvs/edge-feat.csv") as f:
            reader = csv.reader(f)
            edge_feat = [row for row in reader]

        with open("data_generate/data_for_graphormer/raw/original_csvs/graph-label.csv") as f:
            reader = csv.reader(f)
            graph_label = [row for row in reader]

        with open("data_generate/data_for_graphormer/raw/original_csvs/node-feat.csv") as f:
            reader = csv.reader(f)
            node_feat = [row for row in reader]

        with open("data_generate/data_for_graphormer/raw/original_csvs/num-edge-list.csv") as f:
            reader = csv.reader(f)
            num_edge_list = [row for row in reader]

        with open("data_generate/data_for_graphormer/raw/original_csvs/num-node-list.csv") as f:
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

        # 下面这行代码运行报错
        # name = ['one', 'two', 'three']
        test = pd.DataFrame(data=edge)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/edge.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=edge_feat)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/edge-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=graph_label)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/graph-label.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=node_feat)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/node-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=num_edge_list)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/num-edge-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        test = pd.DataFrame(data=num_node_list)  # 数据有三列，列名分别为one,two,three
        # print(test)
        csv_file_name = folder_path + '/num-node-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        


        print("finish create drop_", i,"\n")


if __name__ == '__main__':
    create_csvs()



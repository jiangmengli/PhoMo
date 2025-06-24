# encoding=utf-8
import os
import csv
from config import *
import molecule_data_processing as mdp


def sgp_main_process(chemdraw_file_list_first_substructure, chemdraw_record_version):
    # Get data
    molecule_list = []
    edge_list = []
    edge_type_dict = {}
    for chemdraw_file in chemdraw_file_list_first_substructure:
        sub_molecule_list, sub_edge_list, edge_type_dict = mdp.read_chemdrawfile(chemdraw_file, edge_type_dict)
        molecule_list.append(sub_molecule_list)
        edge_list.append(sub_edge_list)
    # Output records
    if not os.path.isdir('./outputs/'):
        os.mkdir('./outputs/')
    # molecule oriented dataset ----------------------------------------------------------------------------------------
    print('Process: Molecule Oriented Dataset ...')
    output_path = './outputs/' + chemdraw_record_version + '/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_path += 'first_substructure/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # Save graph info
    with open(output_path + 'PC_edge.csv', 'w') as f:
        f_csv = csv.writer(f)
        csv_save = []
        for sub_edge_list in edge_list:
            for edge_dict in sub_edge_list:
                csv_save.append([int(edge_dict['head']) - 1, int(edge_dict['tail']) - 1])
        f_csv.writerows(csv_save)
        f.close()
        f.close()
    invert_edge_type_dict = {}
    for edge_type_i in edge_type_dict:
        invert_edge_type_dict[str(edge_type_dict[edge_type_i])] = edge_type_i
    print('Performing \'edge_feat_getter\' ...')
    with open(output_path + 'PC_edge_feat.csv', 'w') as f:
        f_csv = csv.writer(f)
        csv_save = []
        for i in range(len(edge_list)):
            for edge_dict in edge_list[i]:
                csv_save.append(mdp.edge_feat_getter(edge_dict, invert_edge_type_dict))
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_num_edge_list.csv', 'w') as f:
        f_csv = csv.writer(f)
        csv_save = []
        for i in range(len(edge_list)):
            csv_save.append([int(len(edge_list[i]))])
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_num_node_list.csv', 'w') as f:
        f_csv = csv.writer(f)
        csv_save = []
        for i in range(len(molecule_list)):
            csv_save.append([int(len(molecule_list[i]))])
        f_csv.writerows(csv_save)
        f.close()
    molecule_obj = []
    for i in range(len(molecule_list)):
        for j in range(len(molecule_list[i])):
            molecule_obj.append(molecule_list[i][j][-1])
    molecule_obj = list(set(molecule_obj))
    molecule_check = {}
    for i in range(len(molecule_obj)):
        molecule_check[molecule_obj[i]] = i + 1
    print('Performing \'node_feat_getter\' ...')
    with open(output_path + 'PC_node_feat_detail.csv', 'w') as f:
        f_csv = csv.writer(f)
        csv_save = []
        PC_node_feat_csv_save = []
        for i in range(len(molecule_list)):
            for j in range(len(molecule_list[i])):
                csv_save_add, PC_node_feat_csv_save_add, molecule_list = mdp.node_feat_getter(i, j, molecule_list,
                                                                                              molecule_check,
                                                                                              invert_edge_type_dict,
                                                                                              edge_list)
                csv_save.append(csv_save_add)
                PC_node_feat_csv_save.append(PC_node_feat_csv_save_add)
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_node_feat.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(PC_node_feat_csv_save)
        f.close()
    return True


if __name__ == '__main__':
    # Setting
    chemdraw_record_version = config_chemdraw_record_version
    chemdraw_file_list = ['./inputs/' + chemdraw_record_version + '/graphs/']

    # Processing file list first_substructure
    chemdraw_file_list_first_substructure = []
    if len(chemdraw_file_list) == 1:
        if not chemdraw_file_list[0].endswith('.ct'):
            check_dir = chemdraw_file_list[0] + 'first_substructure/'
            for root, dirs, files in os.walk(check_dir, True):
                for name in files:
                    file_path = str(os.path.join(check_dir, name))
                    if file_path.endswith('.ct') and os.path.exists(file_path):
                        chemdraw_file_list_first_substructure.append([file_path, int(str(name).split(' ')[0])])
    chemdraw_file_list_first_substructure_temp = sorted(chemdraw_file_list_first_substructure, key=lambda ele: ele[-1])
    chemdraw_file_list_first_substructure = []
    for ele in chemdraw_file_list_first_substructure_temp:
        chemdraw_file_list_first_substructure.append(str(ele[0]))
    # Main processes
    if sgp_main_process(chemdraw_file_list_first_substructure, chemdraw_record_version):
        print('Process: ChemDraw_Graph_Processing Finished')
    else:
        print('Error: ChemDraw_Graph_Processing Failed')

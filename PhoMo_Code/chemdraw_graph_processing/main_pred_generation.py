# encoding=utf-8
import os
import csv
import json
import numpy as np
from keras.utils import to_categorical
import math

import utils
import pred_data_processing as pdp
import molecule_data_processing as mdp
from config import *


def cgp_main_process(chemdraw_file_list_type_conjugated, chemdraw_file_list_stereo, chemdraw_record_version):
    # For positive samples -----------------------
    # Get data
    molecule_list = []
    edge_list = []
    edge_type_dict = {}
    for chemdraw_file in chemdraw_file_list_type_conjugated:
        sub_molecule_list, sub_edge_list, edge_type_dict = mdp.read_chemdrawfile(chemdraw_file, edge_type_dict)
        molecule_list.append(sub_molecule_list)
        edge_list.append(sub_edge_list)
    # Get stereo data
    stereo_molecule_list = []
    stereo_edge_list = []
    stereo_edge_type_dict = {}
    for chemdraw_file in chemdraw_file_list_stereo:
        stereo_sub_molecule_list, stereo_sub_edge_list, stereo_edge_type_dict = mdp.read_chemdrawfile(chemdraw_file, stereo_edge_type_dict)
        stereo_molecule_list.append(stereo_sub_molecule_list)
        stereo_edge_list.append(stereo_sub_edge_list)
    # Molecule graph matching check
    if utils.mgm_check(molecule_list, edge_list, stereo_molecule_list, stereo_edge_list):
        print('Process: Positive molecule graph matching checking -> Finished!')
    else:
        print('Process: Positive molecule graph matching checking -> Error...')
        exit()
    # Output records
    if not os.path.isdir('./outputs/'):
        os.mkdir('./outputs/')
    # molecule oriented dataset ----------------------------------------------------------------------------------------
    print('Process: Molecule Oriented Dataset ...')
    output_path = './outputs/' + chemdraw_record_version + '/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # Save graph info
    with open(output_path + 'PC_edge.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        csv_save = []
        for sub_edge_list in edge_list:
            for edge_dict in sub_edge_list:
                if edge_numbering_from_zero:
                    csv_save.append([int(edge_dict['head']) - 1, int(edge_dict['tail']) - 1])
                else:
                    csv_save.append([int(edge_dict['head']), int(edge_dict['tail'])])
        f_csv.writerows(csv_save)
        f.close()
        f.close()
    invert_edge_type_dict = {}
    for edge_type_i in edge_type_dict:
        invert_edge_type_dict[str(edge_type_dict[edge_type_i])] = edge_type_i
    invert_stereo_edge_type_dict = {}
    for stereo_edge_type_i in stereo_edge_type_dict:
        invert_stereo_edge_type_dict[str(stereo_edge_type_dict[stereo_edge_type_i])] = stereo_edge_type_i
    print('Performing \'edge_feat_getter\' ...')
    with open(output_path + 'PC_edge_feat.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        csv_save = []
        for i in range(len(edge_list)):
            for j in range(len(edge_list[i])):
                edge_dict = edge_list[i][j]
                stereo_edge_dict = stereo_edge_list[i][j]
                csv_save.append(mdp.edge_feat_getter(edge_dict, invert_edge_type_dict, stereo_edge_dict=stereo_edge_dict, molecule=(i+1), invert_stereo_edge_type_dict=invert_stereo_edge_type_dict))
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_num_edge_list.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        csv_save = []
        for i in range(len(edge_list)):
            csv_save.append([int(len(edge_list[i]))])
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_num_node_list.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
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
        f_csv = csv.writer(f, lineterminator='\n')
        csv_save = []
        PC_node_feat_csv_save = []
        for i in range(len(molecule_list)):
            for j in range(len(molecule_list[i])):
                csv_save_add, PC_node_feat_csv_save_add, molecule_list = mdp.node_feat_getter(i, j, molecule_list,
                                                                                              molecule_check,
                                                                                              invert_edge_type_dict,
                                                                                              edge_list, stereo_molecule_list=stereo_molecule_list)
                csv_save.append(csv_save_add)
                PC_node_feat_csv_save.append(PC_node_feat_csv_save_add)
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_node_feat.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        f_csv.writerows(PC_node_feat_csv_save)
        f.close()
    # Save graph labels
    molecular_property_list, abv_molecule_list, pred_molecule_key_list, pred_molecule_post_descrip = pdp.read_moleculefile_cataperform(
        molecule_list, edge_list, invert_edge_type_dict)
    with open(output_path + 'PC_molecular_cataperform_json.json', 'w') as f:
        f.write(json.dumps(molecular_property_list))
    abv_molecule_list = sorted(abv_molecule_list, key=lambda ele: ele[0])
    with open(output_path + 'PC_molecular_cataperform.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        csv_save = []
        for i in range(len(abv_molecule_list)):
            for j in range(len(abv_molecule_list[i])):
                abv_molecule_list[i][j] = float(abv_molecule_list[i][j])
            csv_save.append(abv_molecule_list[i])
        f_csv.writerows(csv_save)
        f.close()
    with open(output_path + 'PC_molecular_cataperform_description.txt', 'w') as f:
        f.write(str('Molecule keys:\n' + ', '.join(pred_molecule_key_list)) + '\n\n')
        f.write(str('Molecule key description:\n' + str(pred_molecule_post_descrip) + '\n\n'))
        f.close()
    return True


if __name__ == '__main__':
    # Setting
    chemdraw_record_version = config_chemdraw_pred_version
    chemdraw_file_list = ['./inputs/' + chemdraw_record_version + '/graphs/']

    # For positive samples -----------------------------
    # Processing file list type_conjugated
    if len(chemdraw_file_list) == 1:
        if not chemdraw_file_list[0].endswith('.ct'):
            check_dir = chemdraw_file_list[0] + 'type_conjugated/'
            chemdraw_file_list_type_conjugated = []
            for root, dirs, files in os.walk(check_dir, True):
                for name in files:
                    file_path = str(os.path.join(check_dir, name))
                    if file_path.endswith('.ct') and os.path.exists(file_path):
                        chemdraw_file_list_type_conjugated.append([file_path, int(str(name).split('_')[0])])
    chemdraw_file_list_type_conjugated_temp = sorted(chemdraw_file_list_type_conjugated, key=lambda ele: ele[-1])
    chemdraw_file_list_type_conjugated = []
    for ele in chemdraw_file_list_type_conjugated_temp:
        chemdraw_file_list_type_conjugated.append(str(ele[0]))
    # Processing file list - stereo
    if len(chemdraw_file_list) == 1:
        if not chemdraw_file_list[0].endswith('.ct'):
            check_dir = chemdraw_file_list[0] + 'stereo/'
            chemdraw_file_list_stereo = []
            for root, dirs, files in os.walk(check_dir, True):
                for name in files:
                    file_path = str(os.path.join(check_dir, name))
                    if file_path.endswith('.ct') and os.path.exists(file_path):
                        chemdraw_file_list_stereo.append([file_path, int(str(name).split('_')[0])])
    chemdraw_file_list_stereo_temp = sorted(chemdraw_file_list_stereo, key=lambda ele: ele[-1])
    chemdraw_file_list_stereo = []
    for ele in chemdraw_file_list_stereo_temp:
        chemdraw_file_list_stereo.append(str(ele[0]))
    # Main processes -----------------------------------
    if cgp_main_process(chemdraw_file_list_type_conjugated, chemdraw_file_list_stereo, chemdraw_record_version):
        print('Process: ChemDraw_Graph_Processing Finished')
    else:
        print('Error: ChemDraw_Graph_Processing Failed')

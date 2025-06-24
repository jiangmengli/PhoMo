# encoding=utf-8
import os
import csv
import utils

from config import *


def cr_edge_acc_main_process(chemdraw_record_version, model_type_list):
    # Read data
    source_path = './inputs/' + chemdraw_record_version + '/'
    mol_exp_dict = {}
    with open(source_path + 'graph_labels.csv') as f:
        f_csv = csv.reader(f)
        _ = next(f_csv)
        for row in f_csv:
            mol = str(int(row[0]) - 1)
            if mol in list(mol_exp_dict.keys()):
                mol_exp_dict[mol] += 1
            else:
                mol_exp_dict[mol] = 1
    mol_ct_path = source_path + 'type_conjugated/'
    mol_ct_file_list = []
    for root, dirs, files in os.walk(mol_ct_path, True):
        for name in files:
            file_path = str(os.path.join(mol_ct_path, name))
            if file_path.endswith('.ct') and os.path.exists(file_path):
                mol_ct_file_list.append([file_path, int(str(name).split(' ')[0])])
    # Iteration
    for model_type in model_type_list:
        num_edge_list_temp = []
        with open(source_path + 'num-edge-list.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                num_edge_list_temp.append(int(row[0]))
        model_acc_diff_list = []
        with open(source_path + model_type + '_acc_diffs.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                model_acc_diff_list.append((row))
        # Processing experiment
        num_edge_list = []
        mol_model_acc_diff_list = []
        for mol_i in range(len(list(mol_exp_dict.keys()))):
            mol_acc_diff_list = []
            mol_exp_num = int(mol_exp_dict[str(mol_i)])
            for exp_i in range(mol_exp_num):
                if mol_acc_diff_list:
                    temp = model_acc_diff_list[0].copy()
                    for j in range(len(temp)):
                        mol_acc_diff_list[j] += float(temp[j]) / mol_exp_num
                else:
                    num_edge_list.append(int(num_edge_list_temp[0]))
                    temp = model_acc_diff_list[0].copy()
                    for j in range(len(temp)):
                        mol_acc_diff_list.append(float(temp[j]) / mol_exp_num)
                del model_acc_diff_list[0]
                del num_edge_list_temp[0]
            mol_model_acc_diff_list.append(mol_acc_diff_list)
        del model_acc_diff_list
        del num_edge_list_temp
        # Processing molecule
        mol_edge_causal_classification = []
        for mol_i in range(len(mol_model_acc_diff_list)):
            mol_edge_num = num_edge_list[mol_i]
            j = mol_edge_num
            acc_bound = [0, 0]
            while j < len(mol_model_acc_diff_list[mol_i]):
                temp = float(mol_model_acc_diff_list[mol_i][j])
                if temp < acc_bound[0]:
                    acc_bound[0] = temp
                if temp > acc_bound[1]:
                    acc_bound[1] = temp
                j += 1
            level_acc_bound = {
                'strong': acc_bound[0] + causal_level['strong'],
                'medium': acc_bound[0],
                'weak': acc_bound[1]
            }
            edge_causal_classification = []
            for j in range(mol_edge_num):
                temp = float(mol_model_acc_diff_list[mol_i][j])
                if temp < level_acc_bound['strong']:
                    edge_causal_classification.append('strong')
                elif temp < level_acc_bound['medium']:
                    edge_causal_classification.append('medium')
                elif temp <= level_acc_bound['weak']:
                    edge_causal_classification.append('weak')
                else:
                    edge_causal_classification.append('confounder')
            mol_edge_causal_classification.append(edge_causal_classification)
        # Create path
        output_path = './outputs/' + chemdraw_record_version + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path += model_type + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(output_path + model_type + '_causal_class.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(mol_edge_causal_classification)
            f.close()
        # Remove strong ct
        output_path += 'remove_strong/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'strong':
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'medium':
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove strong_medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_strong_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'strong' or mol_edge_causal_classification[mol_i][skip_i] == 'medium':
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove confounder ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_confounder/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'confounder':
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve strong ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_strong/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'strong':
                    continue
                else:
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'medium':
                    continue
                else:
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve strong_medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_strong_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'strong' or mol_edge_causal_classification[mol_i][skip_i] == 'medium':
                    continue
                else:
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve confounder ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_confounder/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_edge_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            start_line = int(mol_ct[1].split(' ')[0]) + 2
            del_num = 0
            for skip_i in range(len(mol_edge_causal_classification[mol_i])):
                if mol_edge_causal_classification[mol_i][skip_i] == 'confounder':
                    continue
                else:
                    del mol_ct[start_line + skip_i - del_num]
                    del_num += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(
                int(mol_ct[1].split(' ')[1].strip('\n')) - del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)

    return True


def cr_node_acc_main_process(chemdraw_record_version, model_type_list):
    # Read data
    del_stander = config_del_stander
    source_path = './inputs/' + chemdraw_record_version + '/'
    mol_exp_dict = {}
    with open(source_path + 'graph_labels.csv') as f:
        f_csv = csv.reader(f)
        _ = next(f_csv)
        for row in f_csv:
            mol = str(int(row[0]) - 1)
            if mol in list(mol_exp_dict.keys()):
                mol_exp_dict[mol] += 1
            else:
                mol_exp_dict[mol] = 1
    mol_ct_path = source_path + 'type_conjugated/'
    mol_ct_file_list = []
    for root, dirs, files in os.walk(mol_ct_path, True):
        for name in files:
            file_path = str(os.path.join(mol_ct_path, name))
            if file_path.endswith('.ct') and os.path.exists(file_path):
                mol_ct_file_list.append([file_path, int(str(name).split(' ')[0])])
    # Iteration
    for model_type in model_type_list:
        num_node_list_temp = []
        with open(source_path + 'num-node-list.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                num_node_list_temp.append(int(row[0]))
        model_acc_diff_list = []
        with open(source_path + model_type + '_acc_diffs.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                model_acc_diff_list.append((row))
        # Processing experiment
        num_node_list = []
        mol_model_acc_diff_list = []
        for mol_i in range(len(list(mol_exp_dict.keys()))):
            mol_acc_diff_list = []
            mol_exp_num = int(mol_exp_dict[str(mol_i)])
            for exp_i in range(mol_exp_num):
                if mol_acc_diff_list:
                    temp = model_acc_diff_list[0].copy()
                    for j in range(len(temp)):
                        mol_acc_diff_list[j] += float(temp[j]) / mol_exp_num
                else:
                    num_node_list.append(int(num_node_list_temp[0]))
                    temp = model_acc_diff_list[0].copy()
                    for j in range(len(temp)):
                        mol_acc_diff_list.append(float(temp[j]) / mol_exp_num)
                del model_acc_diff_list[0]
                del num_node_list_temp[0]
            mol_model_acc_diff_list.append(mol_acc_diff_list)
        del model_acc_diff_list
        del num_node_list_temp
        # Processing molecule
        mol_node_causal_classification = []
        for mol_i in range(len(mol_model_acc_diff_list)):
            mol_node_num = num_node_list[mol_i]
            j = mol_node_num
            acc_bound = [0, 0]
            while j < len(mol_model_acc_diff_list[mol_i]):
                temp = float(mol_model_acc_diff_list[mol_i][j])
                if temp < acc_bound[0]:
                    acc_bound[0] = temp
                if temp > acc_bound[1]:
                    acc_bound[1] = temp
                j += 1
            level_acc_bound = {
                'strong': acc_bound[0] + causal_level['strong'],
                'medium': acc_bound[0],
                'weak': acc_bound[1]
            }
            node_causal_classification = []
            for j in range(mol_node_num):
                temp = float(mol_model_acc_diff_list[mol_i][j])
                if temp < level_acc_bound['strong']:
                    node_causal_classification.append('strong')
                elif temp < level_acc_bound['medium']:
                    node_causal_classification.append('medium')
                elif temp <= level_acc_bound['weak']:
                    node_causal_classification.append('weak')
                else:
                    node_causal_classification.append('confounder')
            mol_node_causal_classification.append(node_causal_classification)
        # Create path
        output_path = './outputs/' + chemdraw_record_version + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path += model_type + '/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(output_path + model_type + '_causal_class.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(mol_node_causal_classification)
            f.close()
        # Remove strong ct
        output_path += 'remove_strong/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'strong':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'medium':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove strong_medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_strong_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'strong' or mol_node_causal_classification[mol_i][skip_i] == 'medium':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Remove confounder ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'remove_confounder/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'confounder':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve strong ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_strong/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'strong':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'medium':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve strong_medium ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_strong_medium/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'strong' or mol_node_causal_classification[mol_i][skip_i] == 'medium':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)
        # Reserve confounder ct
        output_path = './outputs/' + chemdraw_record_version + '/' + model_type + '/'
        output_path += 'reserve_confounder/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mol_i in range(len(mol_node_causal_classification)):
            mol_ct_file = ''
            for mol_ct_i in range(len(mol_ct_file_list)):
                if int(mol_ct_file_list[mol_ct_i][-1]) == mol_i + 1:
                    mol_ct_file = mol_ct_file_list[mol_ct_i][0]
                    break
            # Read ct file
            mol_ct = utils.read_chemdrawfile(mol_ct_file)
            node_start_line = 2
            edge_del_num = 0
            for skip_i in range(len(mol_node_causal_classification[mol_i])):
                if mol_node_causal_classification[mol_i][skip_i] == 'confounder':
                    mol_ct_temp = mol_ct[node_start_line + skip_i].split(' ')
                    del mol_ct_temp[-1]
                    mol_ct_temp = ' '.join(mol_ct_temp)
                    mol_ct[node_start_line + skip_i] = mol_ct_temp + ' ' + str(del_stander) + '\n'
                    ite_i = int(mol_ct[1].split(' ')[0]) + 2
                    while True:
                        if not mol_ct[ite_i]:
                            break
                        else:
                            temp_mol_text = mol_ct[ite_i].rstrip('\n')
                            temp_mol_text = temp_mol_text.split(' ')
                            if int(temp_mol_text[0]) == int(skip_i) + 1 or int(temp_mol_text[1]) == int(skip_i) + 1:
                                del mol_ct[ite_i]
                                edge_del_num += 1
                            else:
                                ite_i += 1
            mol_ct[1] = mol_ct[1].split(' ')[0] + ' ' + str(int(mol_ct[1].split(' ')[1].strip('\n')) - edge_del_num) + '\n'
            utils.write_chemdrawfile(output_path + mol_ct_file.split('/')[-1], mol_ct)

    return True


if __name__ == '__main__':
    # Setting
    chemdraw_record_version = config_chemdraw_record_version
    model_type_list = ['both', 'GNN', 'graph_search']

    # Main processes -----------------------------------
    if chemdraw_record_version.endswith('Edge'):
        if cr_edge_acc_main_process(chemdraw_record_version, model_type_list):
            print('Process: ChemDraw_Causality_Reasoning-Edge (Accuracy) Finished')
        else:
            print('Error: ChemDraw_Causality_Reasoning-Edge (Accuracy) Failed')
    elif chemdraw_record_version.endswith('Node'):
        if cr_node_acc_main_process(chemdraw_record_version, model_type_list):
            print('Process: ChemDraw_Causality_Reasoning-Node (Accuracy) Finished')
        else:
            print('Error: ChemDraw_Causality_Reasoning-Node (Accuracy) Failed')
    else:
        print('Error: \'chemdraw_record_version\' is INCORRECT!')

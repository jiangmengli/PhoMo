# encoding=utf-8
import numpy as np

import utils
from config import *


def read_chemdrawfile(chemdraw_file, edge_type_dict):
    # Read text
    molecule_num = 0
    molecule_list = []
    edge_list = []
    with open(chemdraw_file, 'r') as file_to_read:
        i = 0
        while True:
            i += 1
            lines = file_to_read.readline().strip(' ').strip('\n')
            if not lines:
                break
                pass
            if i == 1:
                pass
                continue
            elif i == 2:
                molecule_num = int(lines.split(' ')[0])
            else:
                if i < molecule_num + 3:
                    add_molecule = lines.split(' ')
                    subm = 0
                    len_subm = len(add_molecule)
                    while subm < len_subm:
                        add_molecule[subm].strip()
                        if add_molecule[subm]:
                            subm += 1
                        else:
                            len_subm -= 1
                            del add_molecule[subm]
                    molecule_list.append(add_molecule)
                else:
                    lines = lines.strip(' ').split(' ')
                    k = 0
                    k_len = len(lines)
                    while k < k_len:
                        if not lines[k]:
                            del lines[k]
                            k_len -= 1
                        else:
                            k += 1
                    edge_type = str(lines[-2]) + ' ' + str(lines[-1])
                    if not edge_type in edge_type_dict:
                        edge_type_dict[edge_type] = len(list(edge_type_dict.keys()))
                    edge_type = edge_type_dict[edge_type]
                    edge_dict = {'head': int(lines[0]), 'tail': int(lines[1]), 'type': int(edge_type)}
                    edge_list.append(edge_dict)
            pass
        pass
    return molecule_list, edge_list, edge_type_dict


def edge_feat_getter(edge_dict, invert_edge_type_dict, stereo_edge_dict=None, molecule=None, invert_stereo_edge_type_dict=None):
    edge_type_detail = int(edge_dict['type'])
    possible_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
    possible_is_conjugated = utils.bond_conjugated_detection(invert_edge_type_dict[str(edge_type_detail)])

    if possible_is_conjugated:
        possible_is_conjugated = 1
    else:
        possible_is_conjugated = 0

    if stereo_edge_dict:
        possible_bond_stereo = utils.bond_stereo_detection(invert_stereo_edge_type_dict[str(stereo_edge_dict['type'])])
    else:
        possible_bond_stereo = 'STEREONONE'

    # Molecular edge attribute rearrangement
    if possible_bond_type == 'SINGLE':
        possible_bond_type = 0
    elif possible_bond_type == 'DOUBLE':
        possible_bond_type = 1
    elif possible_bond_type == 'TRIPLE':
        possible_bond_type = 2
    elif possible_bond_type == 'AROMATIC':
        possible_bond_type = 3
    else: # 'misc'
        possible_bond_type = 4

    if possible_bond_stereo == 'STEREONONE':
        possible_bond_stereo = 0
    elif possible_bond_stereo == 'STEREOZ':
        possible_bond_stereo = 1
    elif possible_bond_stereo == 'STEREOE':
        possible_bond_stereo = 2
    elif possible_bond_stereo == 'STEREOCIS':
        possible_bond_stereo = 3
    elif possible_bond_stereo == 'STEREOTRANS':
        possible_bond_stereo = 4
    else: # 'STEREOANY'
        possible_bond_stereo = 5

    return [edge_type_detail, possible_bond_type, possible_bond_stereo, possible_is_conjugated]


def node_feat_getter(i, j, molecule_list, molecule_check, invert_edge_type_dict, edge_list, stereo_molecule_list=None):
    node_j = j + 1
    # Pre process
    atom_name = str(molecule_list[i][j][-1])
    if atom_name.endswith('+'):
        possible_formal_charge = 1
        atom_name = atom_name[:-1]
    elif atom_name.endswith('-'):
        possible_formal_charge = -1
        atom_name = atom_name[:-1]
    else:
        possible_formal_charge = 0
    atom_id = int(molecule_check[atom_name])

    possible_numH = utils.get_possible_numH(atom_attributes, atom_name, invert_edge_type_dict, edge_list[i],
                                            molecule_list[i], node_j)

    node_list = [molecule_list[i][j].copy()[:-1]]
    for edgenode in edge_list[i]:
        if node_j == int(edgenode['head']):
            node_list.append(molecule_list[i][int(edgenode['tail']) - 1].copy()[:-1])
        elif node_j == int(edgenode['tail']):
            node_list.append(molecule_list[i][int(edgenode['head']) - 1].copy()[:-1])
    average_cos = []
    for iteratea in range(1, len(node_list)):
        for iterateb in range(iteratea + 1, len(node_list)):
            nodea = node_list[iteratea]
            nodeb = node_list[iterateb]
            nodeo = node_list[0]
            linea = (float(nodeo[0]) - float(nodea[0])) ** 2 + (float(nodeo[1]) - float(nodea[1])) ** 2 + (
                        float(nodeo[2]) - float(nodea[2])) ** 2
            lineb = (float(nodeo[0]) - float(nodeb[0])) ** 2 + (float(nodeo[1]) - float(nodeb[1])) ** 2 + (
                        float(nodeo[2]) - float(nodeb[2])) ** 2
            linec = (float(nodea[0]) - float(nodeb[0])) ** 2 + (float(nodea[1]) - float(nodeb[1])) ** 2 + (
                    float(nodea[2]) - float(nodeb[2])) ** 2
            average_cos.append((linea + lineb - linec) / (2 * (linea ** 0.5) * (lineb ** 0.5)))
    if average_cos:
        average_cos = np.mean(average_cos)
        for iteratei in range(len(cos_check_list)):
            if average_cos >= cos_check_list[iteratei]:
                average_cos = iteratei + 1
                break
        if type(average_cos) is float:
            average_cos = len(cos_check_list) + 1
    else:
        average_cos = 0

    # All atom have NO chirality
    possible_chirality = 'CHI_UNSPECIFIED'

    if stereo_molecule_list:
        hybridization_atom_name = str(stereo_molecule_list[i][j][-1])
        if hybridization_atom_name in stereo_replacement_dict['SP']:
            possible_hybridization = 'SP'
        elif hybridization_atom_name in stereo_replacement_dict['SP2']:
            possible_hybridization = 'SP2'
        elif hybridization_atom_name in stereo_replacement_dict['SP3']:
            possible_hybridization = 'SP3'
        elif hybridization_atom_name in stereo_replacement_dict['SP3D']:
            possible_hybridization = 'SP3D'
        elif hybridization_atom_name in stereo_replacement_dict['SP3D2']:
            possible_hybridization = 'SP3D2'
        else:
            # hypotest = ['Fe', 'Br', 'H', 'Cl', 'F', 'C']
            # if hybridization_atom_name == 'C' or hybridization_atom_name == 'N':
            #     print('Molecule: ' + str(i) + '; Atom Num: ' + str(len(stereo_molecule_list[i])) + '; Atom Numbering: ' + str(j))
            # if not hybridization_atom_name in hypotest:
            #     print(hybridization_atom_name)
            possible_hybridization = 'misc'
    else:
        possible_hybridization = 'misc'

    possible_is_aromatic = False
    for edgenode in edge_list[i]:
        if node_j == int(edgenode['head']) or node_j == int(edgenode['tail']):
            if utils.bond_type_detection(invert_edge_type_dict[str(edgenode['type'])]) == 'AROMATIC':
                possible_is_aromatic = True
                break

    possible_is_in_ring = possible_is_aromatic
    if not possible_is_in_ring:
        possible_is_in_ring = utils.find_ring(edge_list[i], [node_j].copy())
    if possible_is_in_ring:
        if len(node_list) < 3:
            possible_is_in_ring = False

    if possible_is_aromatic:
        possible_is_aromatic = 1
    else:
        possible_is_aromatic = 0
    if possible_is_in_ring:
        possible_is_in_ring = 1
    else:
        possible_is_in_ring = 0

    # Molecular node attribute rearrangement
    if possible_chirality == 'CHI_UNSPECIFIED':
        possible_chirality = 0
    elif possible_chirality == 'CHI_TETRAHEDRAL_CW':
        possible_chirality = 1
    elif possible_chirality == 'CHI_TETRAHEDRAL_CCW':
        possible_chirality = 2
    else: # 'CHI_OTHER'
        possible_chirality = 3

    if possible_hybridization == 'SP':
        possible_hybridization = 0
    elif possible_hybridization == 'SP2':
        possible_hybridization = 1
    elif possible_hybridization == 'SP3':
        possible_hybridization = 2
    elif possible_hybridization == 'SP3D':
        possible_hybridization = 3
    elif possible_hybridization == 'SP3D2':
        possible_hybridization = 4
    else: # 'misc'
        possible_hybridization = 5

    possible_formal_charge = possible_formal_charge + 5

    csv_save_add = [atom_id, atom_name,
         atom_attributes[atom_name]['possible_atomic_num'], possible_chirality,
         atom_attributes[atom_name]['possible_degree'], possible_formal_charge, possible_numH,
         atom_attributes[atom_name]['possible_number_radical_e'], possible_hybridization,
         possible_is_aromatic, possible_is_in_ring, average_cos,
         molecule_list[i][j][0], molecule_list[i][j][1], molecule_list[i][j][2]]
    PC_node_feat_csv_save_add = [atom_attributes[atom_name]['possible_atomic_num'], possible_chirality,
         atom_attributes[atom_name]['possible_degree'], possible_formal_charge, possible_numH,
         atom_attributes[atom_name]['possible_number_radical_e'], possible_hybridization,
         possible_is_aromatic, possible_is_in_ring, average_cos]
    molecule_list[i][j][-1] = atom_name
    return csv_save_add, PC_node_feat_csv_save_add, molecule_list

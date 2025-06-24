# encoding=utf-8
from config import *


def find_ring(edgei, node_edge_list):
    search = node_edge_list[-1]
    if len(node_edge_list) > 1:
        check_list = node_edge_list.copy()
        check_list = check_list[1:-1]
    else:
        check_list = []
    for edgenode in edgei:
        if check_list:
            if int(edgenode['head']) in check_list or int(edgenode['tail']) in check_list:
                continue
        else:
            if int(edgenode['head']) in node_edge_list and int(edgenode['tail']) in node_edge_list:
                continue
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[0]:
                return True
            else:
                node_edge_list.append(int(edgenode['tail']))
                if find_ring(edgei, node_edge_list.copy()):
                    return True
        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[0]:
                return True
            else:
                node_edge_list.append(int(edgenode['head']))
                if find_ring(edgei, node_edge_list.copy()):
                    return True
    return False


def check_alhyl_detail(check_node, check_type, node_edge_list, target_cata_node, invert_edge_type_dict):
    if check_node in node_edge_list:
        return False
    else:
        if not str(target_cata_node[check_node - 1][-1]) in ['C', 'H']:
            return False
        else:
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(check_type)])
            if not edgej_bond_type == 'SINGLE':
                return False
            else:
                return True


def check_alhyl(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict):
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            if check_alhyl_detail(int(edgenode['tail']), int(edgenode['type']), node_edge_list, target_cata_node,
                                  invert_edge_type_dict):
                node_edge_list.append(int(edgenode['tail']))
                if check_alhyl(node_edge_list.copy(), target_cata_edge, target_cata_node, invert_edge_type_dict):
                    return True
                else:
                    return False
            else:
                return False
        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            if check_alhyl_detail(int(edgenode['head']), int(edgenode['type']), node_edge_list, target_cata_node,
                                  invert_edge_type_dict):
                node_edge_list.append(int(edgenode['head']))
                if check_alhyl(node_edge_list.copy(), target_cata_edge, target_cata_node, invert_edge_type_dict):
                    return True
                else:
                    return False
            else:
                return False
    return True


def check_phenyl_detail(check_node, check_type, node_edge_list, target_cata_node, invert_edge_type_dict):
    restrict_list = [node_edge_list.copy()[0]] + node_edge_list.copy()[2:]
    if check_node in restrict_list:
        return False
    else:
        if not str(target_cata_node[check_node - 1][-1]) in ['C', 'H']:
            return False
        else:
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(check_type)])
            if edgej_bond_type == 'SINGLE' and str(target_cata_node[check_node - 1][-1]) == 'H':
                return True
            elif edgej_bond_type == 'AROMATIC' and str(target_cata_node[check_node - 1][-1]) == 'C':
                return True
            else:
                return False


def check_phenyl(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict, edge_type=0):
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type = aromatic_iterate_check(edge_type, invert_edge_type_dict, edgenode)
            if not edge_type:
                return False
            if check_phenyl_detail(int(edgenode['tail']), int(edgenode['type']), node_edge_list, target_cata_node,
                                   invert_edge_type_dict):
                if int(edgenode['tail']) == node_edge_list[1]:
                    return True
                node_edge_list.append(int(edgenode['tail']))
                if check_phenyl(node_edge_list.copy(), target_cata_edge, target_cata_node, invert_edge_type_dict,
                                edge_type=edge_type):
                    return True
                else:
                    return False
            else:
                return False

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type = aromatic_iterate_check(edge_type, invert_edge_type_dict, edgenode)
            if not edge_type:
                return False
            if check_phenyl_detail(int(edgenode['head']), int(edgenode['type']), node_edge_list, target_cata_node,
                                   invert_edge_type_dict):
                if int(edgenode['head']) == node_edge_list[1]:
                    return True
                node_edge_list.append(int(edgenode['head']))
                if check_phenyl(node_edge_list.copy(), target_cata_edge, target_cata_node, invert_edge_type_dict,
                                edge_type=edge_type):
                    return True
                else:
                    return False
            else:
                return False
    return True


def check_phenyl_x_detail(check_node, check_type, node_edge_list, target_cata_node, invert_edge_type_dict):
    sub_x_check = False
    restrict_list = [node_edge_list.copy()[0]] + node_edge_list.copy()[2:]
    if check_node in restrict_list:
        return False, sub_x_check
    else:
        if not str(target_cata_node[check_node - 1][-1]) in ['C', 'H', 'F', 'Cl', 'Br']:
            return False, sub_x_check
        else:
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(check_type)])
            if edgej_bond_type == 'SINGLE':
                if str(target_cata_node[check_node - 1][-1]) == 'H':
                    return True, sub_x_check
                elif str(target_cata_node[check_node - 1][-1]) in ['F', 'Cl', 'Br']:
                    sub_x_check = True
                    return True, sub_x_check
                else:
                    return False, sub_x_check
            elif edgej_bond_type == 'AROMATIC' and str(target_cata_node[check_node - 1][-1]) == 'C':
                return True, sub_x_check
            else:
                return False, sub_x_check


def check_phenyl_x(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict, edge_type=0,
                   x_check=False):
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type = aromatic_iterate_check(edge_type, invert_edge_type_dict, edgenode)
            if not edge_type:
                return False, x_check
            detail_check, sub_x_check = check_phenyl_x_detail(int(edgenode['tail']), int(edgenode['type']),
                                                              node_edge_list, target_cata_node, invert_edge_type_dict)
            if x_check or sub_x_check:
                x_check = True
            else:
                x_check = False
            if detail_check:
                if int(edgenode['tail']) == node_edge_list[1]:
                    return True, x_check
                node_edge_list.append(int(edgenode['tail']))
                phenyl_check, x_check = check_phenyl_x(node_edge_list.copy(), target_cata_edge, target_cata_node,
                                                       invert_edge_type_dict, edge_type=edge_type, x_check=x_check)
                return phenyl_check, x_check
            else:
                return False, x_check

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type = aromatic_iterate_check(edge_type, invert_edge_type_dict, edgenode)
            if not edge_type:
                return False, x_check
            detail_check, sub_x_check = check_phenyl_x_detail(int(edgenode['head']), int(edgenode['type']),
                                                              node_edge_list, target_cata_node, invert_edge_type_dict)
            if x_check or sub_x_check:
                x_check = True
            else:
                x_check = False
            if detail_check:
                if int(edgenode['head']) == node_edge_list[1]:
                    return True, x_check
                node_edge_list.append(int(edgenode['head']))
                phenyl_check, x_check = check_phenyl_x(node_edge_list.copy(), target_cata_edge, target_cata_node,
                                                       invert_edge_type_dict, edge_type=edge_type, x_check=x_check)
                return phenyl_check, x_check
            else:
                return False, x_check
    return True, x_check


def check_cooh_h(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict):
    oh_check = False
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['tail']) - 1][-1]) == 'H' and oh_check == False:
                oh_check = True
            else:
                return False

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['head']) - 1][-1]) == 'H' and oh_check == False:
                oh_check = True
            else:
                return False

        else:
            continue

    # h could be hidden
    return True


def check_cooh(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict):
    o_two_check = False
    oh_check = False
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if str(target_cata_node[int(edgenode['tail']) - 1][-1]) == 'O':
                if edgej_bond_type == 'DOUBLE':
                    o_two_check = True
                elif edgej_bond_type == 'SINGLE':
                    check_list = [search, int(edgenode['tail'])]
                    oh_check = check_cooh_h(check_list.copy(), target_cata_edge, target_cata_node,
                                            invert_edge_type_dict)
                else:
                    return False
            else:
                return False

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if str(target_cata_node[int(edgenode['head']) - 1][-1]) == 'O':
                if edgej_bond_type == 'DOUBLE':
                    o_two_check = True
                elif edgej_bond_type == 'SINGLE':
                    check_list = [search, int(edgenode['head'])]
                    oh_check = check_cooh_h(check_list.copy(), target_cata_edge, target_cata_node,
                                            invert_edge_type_dict)
                else:
                    return False
            else:
                return False

        else:
            continue

    if o_two_check and oh_check:
        return True
    else:
        return False


def check_nh3(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict):
    h_check = 0
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['tail']) - 1][-1]) == 'H':
                h_check += 1
            else:
                return False

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['head']) - 1][-1]) == 'H':
                h_check += 1
            else:
                return False

        else:
            continue

    # h could be hidden
    if h_check < 3:
        return True
    else:
        return False


def check_oh(node_edge_list, target_cata_edge, target_cata_node, invert_edge_type_dict):
    h_check = 0
    search = node_edge_list[-1]
    for edgenode in target_cata_edge:
        if search == int(edgenode['head']):
            if int(edgenode['tail']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['tail']) - 1][-1]) == 'H':
                h_check += 1
            else:
                return False

        elif search == int(edgenode['tail']):
            if int(edgenode['head']) == node_edge_list[-2]:
                continue
            edge_type_detail = int(edgenode['type'])
            edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
            if edgej_bond_type == 'SINGLE' and str(
                    target_cata_node[int(edgenode['head']) - 1][-1]) == 'H':
                h_check += 1
            else:
                return False

        else:
            continue

    # h could be hidden
    if h_check < 2:
        return True
    else:
        return False


def get_possible_numH(atom_attributes, atom_name, invert_edge_type_dict, target_edge_list, target_node_list,
                      index_node):
    possible_numH = int(atom_attributes[atom_name]['possible_numH'])
    if possible_numH:
        for edge_j in range(len(target_edge_list)):
            edgenode = target_edge_list[edge_j]
            if index_node == int(edgenode['head']) or index_node == int(edgenode['tail']):
                check_H = False
                if index_node == int(edgenode['head']):
                    if str(target_node_list[int(edgenode['tail']) - 1][-1]) == 'H':
                        check_H = True
                elif index_node == int(edgenode['tail']):
                    if str(target_node_list[int(edgenode['head']) - 1][-1]) == 'H':
                        check_H = True
                if check_H:
                    possible_numH = 1
                    for edge_k in range(len(target_edge_list)):
                        edgenode_sub = target_edge_list[edge_k]
                        if edge_k == edge_j:
                            continue
                        else:
                            if index_node == int(edgenode_sub['head']):
                                if str(target_node_list[int(edgenode_sub['tail']) - 1][-1]) == 'H':
                                    possible_numH += 1
                            elif index_node == int(edgenode_sub['tail']):
                                if str(target_node_list[int(edgenode_sub['head']) - 1][-1]) == 'H':
                                    possible_numH += 1
                    break
                else:
                    edge_type_detail = int(edgenode['type'])
                    edgej_bond_type = bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'DOUBLE':
                        possible_numH -= 2
                    elif edgej_bond_type == 'AROMATIC':
                        if invert_edge_type_dict[str(edge_type_detail)] == '1 9':
                            possible_numH -= 1
                        elif invert_edge_type_dict[str(edge_type_detail)] == '2 64' or invert_edge_type_dict[str(
                                edge_type_detail)] == '2 65' or invert_edge_type_dict[str(edge_type_detail)] == '2 66':
                            possible_numH -= 2
                    elif edgej_bond_type == 'TRIPLE':
                        possible_numH -= 3
                    else:
                        possible_numH -= 1
    possible_numH = max(int(possible_numH), 0)
    return possible_numH


def mgm_check(check_molecule_list, check_edge_list, molecule_list, edge_list):
    # Check atom
    '''
    for i in range(len(molecule_list)):
        for j in range(len(molecule_list[i])):
            if not molecule_list[i][j][-1] == check_molecule_list[i][j][-1]:
                print('molecule error no: ' + str(i + 1))
                print('atom error no: ' + str(j + 1))
                print('details: ' + str(molecule_list[i][j][-1]) + '!=' + str(check_molecule_list[i][j][-1]))
                return False
    '''
    # Check edge
    check_false = True
    for i in range(len(edge_list)):
        for j in range(len(edge_list[i])):
            if not edge_list[i][j]['head'] == check_edge_list[i][j]['head']:
                print('molecule error no: ' + str(i + 1))
                print('edge error no: ' + str(j + 1))
                print('details: ' + str(edge_list[i][j]['head']) + '!=' + str(check_edge_list[i][j]['head']))
                check_false = False
            if not edge_list[i][j]['tail'] == check_edge_list[i][j]['tail']:
                print('molecule error no: ' + str(i + 1))
                print('edge error no: ' + str(j + 1))
                print('details: ' + str(edge_list[i][j]['tail']) + '!=' + str(check_edge_list[i][j]['tail']))
                check_false = False
    return check_false


# ------------------------------------------ bond type detection -------------------------------------------------------
def bond_type_detection(edge_type):
    if edge_type == '2 0' or edge_type == '2 1' or edge_type == '2 2' or edge_type == '2 8' or edge_type == '2 9' or edge_type == '1 2' or edge_type == '2 4' or edge_type == '2 32' or edge_type == '2 33':
        edge_bond_type = 'DOUBLE'
    elif edge_type == '1 9' or edge_type == '2 64' or edge_type == '2 65' or edge_type == '2 66':
        edge_bond_type = 'AROMATIC'
    elif edge_type == '3 1':
        edge_bond_type = 'TRIPLE'
    else:
        edge_bond_type = 'SINGLE'
    return edge_bond_type


def bond_conjugated_detection(edge_type):
    if edge_type == '1 9' or edge_type == '1 4' or edge_type == '2 64' or edge_type == '2 65' or edge_type == '2 66' or edge_type == '1 2' or edge_type == '2 4' or edge_type == '2 32' or edge_type == '2 33':
        possible_is_conjugated = True
    else:
        possible_is_conjugated = False
    return possible_is_conjugated


def aromatic_iterate_check(edge_type, invert_edge_type_dict, edgenode):
    if edge_type == 1 and invert_edge_type_dict[str(int(edgenode['type']))] == '1 9':
        return False
    elif edge_type == 2:
        if invert_edge_type_dict[str(int(edgenode['type']))] == '2 64' or invert_edge_type_dict[str(
                int(edgenode['type']))] == '2 65' or invert_edge_type_dict[
            str(int(edgenode['type']))] == '2 66':
            return False
    if invert_edge_type_dict[str(int(edgenode['type']))] == '1 9':
        return_edge_type = 1
    elif invert_edge_type_dict[str(int(edgenode['type']))] == '2 64' or invert_edge_type_dict[str(
            int(edgenode['type']))] == '2 65' or invert_edge_type_dict[str(int(edgenode['type']))] == '2 66':
        return_edge_type = 2
    elif invert_edge_type_dict[str(int(edgenode['type']))] == '1 4' or invert_edge_type_dict[
        str(int(edgenode['type']))] == '1 1':
        return_edge_type = 3
    else:
        return False
    return return_edge_type


def bond_stereo_detection(edge_type):
    if edge_type == '2 4' or edge_type == '2 32' or edge_type == '2 33':
        possible_is_stereo = 'STEREOE'
    else:
        possible_is_stereo = 'STEREONONE'
    return possible_is_stereo
# ----------------------------------------------------------------------------------------------------------------------

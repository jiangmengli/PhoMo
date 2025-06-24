# encoding=utf-8
import csv
import math

import utils
from config import *


def read_moleculefile_cataperform(molecule_list, edge_list, invert_edge_type_dict):
    print('Performing \'read_moleculefile_cataperform\' ...')
    # Read csv
    molecular_property_list = []
    abv_molecule_list = []

    for mol_i in range(len(molecule_list)):
        mol_property_dict = {'Catalyst': mol_i + 1}
        try:
            add_abv_list = [int(mol_property_dict['Catalyst'])]
        except:
            continue

        # Catalyst graph analysis
        target_cata_node = molecule_list[int(mol_property_dict['Catalyst']) - 1]
        target_cata_edge = edge_list[int(mol_property_dict['Catalyst']) - 1]

        # Get graph scan
        double_metal_core, metal_core_type, electron_donating_type, electron_withdrawing_type, proton_donor_type = graph_scan(
            target_cata_node, target_cata_edge, invert_edge_type_dict)

        mol_property_dict['Double Metal Core'] = int(double_metal_core)
        mol_property_dict['Metal Core Type'] = int(metal_core_type)
        mol_property_dict['Electron Donating Group'] = int(electron_donating_type)
        mol_property_dict['Electron Withdrawing Group'] = int(electron_withdrawing_type)
        mol_property_dict['Proton Donor'] = int(proton_donor_type)
        add_abv_list.append(int(double_metal_core))
        add_abv_list.append(int(metal_core_type))
        add_abv_list.append(int(electron_donating_type))
        add_abv_list.append(int(electron_withdrawing_type))
        add_abv_list.append(int(proton_donor_type))
        # Save
        molecular_property_list.append(mol_property_dict)
        abv_molecule_list.append(add_abv_list)

    return molecular_property_list, abv_molecule_list, pred_molecule_key_list, pred_molecule_post_descrip

def graph_scan(target_cata_node, target_cata_edge, invert_edge_type_dict):
    # 核数, 中心金属原子普通或贵, 给电子基团, 吸电子基团, 质子源
    # core_number
    normal_metal_count = 0
    noble_metal_count = 0
    for nodei in target_cata_node:
        atom_name = str(nodei[-1])
        if atom_name.endswith('+'):
            atom_name = atom_name[:-1]
        elif atom_name.endswith('-'):
            atom_name = atom_name[:-1]
        if atom_name in normal_metal_atom_list:
            normal_metal_count += 1
        elif atom_name in normal_metal_atom_list:
            noble_metal_count += 1
    double_metal_core = 1 if (normal_metal_count + noble_metal_count) > 1 else 0

    # metal_core_type: 0 -> no metal core; 1 -> all normal; 2 -> all noble; 3 -> both noble and normal
    if noble_metal_count > 0:
        if normal_metal_count > 0:
            metal_core_type = 3
        else:
            metal_core_type = 2
    else:
        if normal_metal_count > 0:
            metal_core_type = 1
        else:
            metal_core_type = 0

    # electron-donating group (给电子基团)
    # electron_donating_type: 0 -> none; 1 -> weak; 2 -> medium; 3 -> strong
    electron_donating_type = 0
    for itei in range(len(target_cata_node)):
        nodei = target_cata_node[itei]
        index_i = itei + 1
        atom_name = str(nodei[-1])
        if atom_name.endswith('+'):
            atom_name = atom_name[:-1]
        elif atom_name.endswith('-'):
            atom_name = atom_name[:-1]
        # Strong
        # O-
        if str(nodei[-1]) == 'O-':
            electron_donating_type = 3
            break
        # NR2, NHR, NH2
        if str(nodei[-1]) == 'N':
            good_num = 0
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            check_list = [index_i, int(edgej['tail'])]
                        else:
                            check_list = [index_i, int(edgej['head'])]
                        # check alkyl
                        check_flag = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                       invert_edge_type_dict)
                        if check_flag:
                            good_num += 1
            if good_num > 1:
                electron_donating_type = 3
                break
            else:
                possible_numH = utils.get_possible_numH(atom_attributes, atom_name, invert_edge_type_dict,
                                                        target_cata_edge, target_cata_node, index_i)
                if possible_numH + good_num > 1:
                    electron_donating_type = 3
                    break
        # OH, OR
        if str(nodei[-1]) == 'O':
            good_num = 0
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            check_list = [index_i, int(edgej['tail'])]
                        else:
                            check_list = [index_i, int(edgej['head'])]
                        # check alkyl
                        check_flag = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                       invert_edge_type_dict)
                        if check_flag:
                            good_num += 1
            if good_num > 1:
                electron_donating_type = 3
                break
            else:
                possible_numH = utils.get_possible_numH(atom_attributes, atom_name, invert_edge_type_dict,
                                                        target_cata_edge, target_cata_node, index_i)
                if possible_numH + good_num > 1:
                    electron_donating_type = 3
                    break
        # Medium
        # NHCOR
        if str(nodei[-1]) == 'N':
            # Check C
            index_c = 0
            check_flag = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                                check_flag = True
                                index_c = int(edgej['tail'])
                                break
                        else:
                            if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                                check_flag = True
                                index_c = int(edgej['head'])
                                break
            if check_flag:
                # Check H
                check_flag = False
                for edgej in target_cata_edge:
                    if edgej['head'] == index_i or edgej['tail'] == index_i:
                        edge_type_detail = int(edgej['type'])
                        edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                        if edgej_bond_type == 'SINGLE':
                            if edgej['head'] == index_i:
                                if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'H':
                                    check_flag = True
                                    break
                            else:
                                if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'H':
                                    check_flag = True
                                    break
                if not check_flag:
                    if utils.get_possible_numH(atom_attributes, atom_name, invert_edge_type_dict, target_cata_edge,
                                               target_cata_node, index_i) > 0:
                        check_flag = True
                if check_flag:
                    # Check O
                    check_flag = False
                    for edgej in target_cata_edge:
                        if edgej['head'] == index_c or edgej['tail'] == index_c:
                            edge_type_detail = int(edgej['type'])
                            edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                            if edgej_bond_type == 'DOUBLE':
                                if edgej['head'] == index_c:
                                    if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                                        check_flag = True
                                        break
                                else:
                                    if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                                        check_flag = True
                                        break
                    if check_flag:
                        # Check R
                        check_flag = False
                        for edgej in target_cata_edge:
                            if edgej['head'] == index_c or edgej['tail'] == index_c:
                                edge_type_detail = int(edgej['type'])
                                edgej_bond_type = utils.bond_type_detection(
                                    invert_edge_type_dict[str(edge_type_detail)])
                                if edgej_bond_type == 'SINGLE':
                                    if edgej['head'] == index_c and str(
                                            target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                                        check_list = [index_c, int(edgej['tail'])]
                                    elif edgej['tail'] == index_c and str(
                                            target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                                        check_list = [index_c, int(edgej['head'])]
                                    else:
                                        check_list = []
                                    if check_list:
                                        # check alkyl
                                        check_flag = utils.check_alhyl(check_list.copy(), target_cata_edge,
                                                                       target_cata_node,
                                                                       invert_edge_type_dict)
                                        if check_flag:
                                            break
                        if check_flag:
                            if electron_donating_type < 2:
                                electron_donating_type = 2
                            break
        # OCOR
        if str(nodei[-1]) == 'C':
            # Check Single and Double O
            check_do_flag = False
            check_so_flag = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])

                    if edgej['head'] == index_i:
                        if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            if edgej_bond_type == 'SINGLE':
                                check_so_flag = True
                            if edgej_bond_type == 'DOUBLE':
                                check_do_flag = True
                            if check_so_flag and check_do_flag:
                                break
                    else:
                        if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            if edgej_bond_type == 'SINGLE':
                                check_so_flag = True
                            if edgej_bond_type == 'DOUBLE':
                                check_do_flag = True
                            if check_so_flag and check_do_flag:
                                break
            if check_so_flag and check_do_flag:
                # Check R
                check_flag = False
                for edgej in target_cata_edge:
                    if edgej['head'] == index_i or edgej['tail'] == index_i:
                        edge_type_detail = int(edgej['type'])
                        edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                        if edgej_bond_type == 'SINGLE':
                            if edgej['head'] == index_i and str(
                                    target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['tail'])]
                            elif edgej['tail'] == index_i and str(
                                    target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['head'])]
                            else:
                                check_list = []
                            if check_list:
                                # check alkyl
                                check_flag = utils.check_alhyl(check_list.copy(), target_cata_edge,
                                                               target_cata_node,
                                                               invert_edge_type_dict)
                                if check_flag:
                                    break
                if check_flag:
                    if electron_donating_type < 2:
                        electron_donating_type = 2
                    break
        # Weak
        # R, CH3
        # Check R
        check_flag = False
        for edgej in target_cata_edge:
            if edgej['head'] == index_i or edgej['tail'] == index_i:
                edge_type_detail = int(edgej['type'])
                edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                if edgej_bond_type == 'SINGLE':
                    if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['tail'])]
                    elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['head'])]
                    else:
                        check_list = []
                    if check_list:
                        # check alkyl
                        check_flag = utils.check_alhyl(check_list.copy(), target_cata_edge,
                                                       target_cata_node,
                                                       invert_edge_type_dict)
                        if check_flag:
                            break
        if check_flag:
            if electron_donating_type < 1:
                electron_donating_type = 1
            break
        # C6H5
        check_flag = False
        for edgej in target_cata_edge:
            if edgej['head'] == index_i or edgej['tail'] == index_i:
                edge_type_detail = int(edgej['type'])
                edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                if edgej_bond_type == 'SINGLE':
                    if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['tail'])]
                    elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['head'])]
                    else:
                        check_list = []
                    if check_list:
                        # check phenyl
                        phenyl_check, x_check = utils.check_phenyl_x(check_list.copy(), target_cata_edge,
                                                                     target_cata_node, invert_edge_type_dict)
                        if phenyl_check:
                            if not x_check:
                                check_flag = True
                                break
        if check_flag:
            if electron_donating_type < 1:
                electron_donating_type = 1
            break
        # CH2COOH
        check_flag = True
        alt_check_list = []
        edge_num_check = 0
        for edgej in target_cata_edge:
            if edgej['head'] == index_i or edgej['tail'] == index_i:
                edge_type_detail = int(edgej['type'])
                edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                if edgej_bond_type == 'SINGLE':
                    if edgej['head'] == index_i:
                        if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                            alt_check_list.append(int(edgej['tail']))
                            edge_num_check += 1
                        elif not str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'H':
                            edge_num_check += 1
                    elif edgej['tail'] == index_i:
                        if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                            alt_check_list.append(int(edgej['head']))
                            edge_num_check += 1
                        elif not str(target_cata_node[int(edgej['head']) - 1][-1]) == 'H':
                            edge_num_check += 1
                    if not edge_num_check == 2:
                        check_flag = False
                        break
                else:
                    check_flag = False
                    break
        if check_flag:
            for alt_check_ele in alt_check_list:
                check_list = [index_i, alt_check_ele]
                check_flag = utils.check_cooh(check_list.copy(), target_cata_edge, target_cata_node,
                                              invert_edge_type_dict)
            if check_flag:
                if electron_donating_type < 1:
                    electron_donating_type = 1
                break

    # electron-withdrawing group
    # electron_withdrawing_type: 0 -> none; 1 -> exists
    electron_withdrawing_type = 0
    for itei in range(len(target_cata_node)):
        nodei = target_cata_node[itei]
        index_i = itei + 1
        atom_name = str(nodei[-1])
        if atom_name.endswith('+'):
            atom_name = atom_name[:-1]
        elif atom_name.endswith('-'):
            atom_name = atom_name[:-1]
        # N+
        if str(nodei[-1]) == 'N+':
            electron_withdrawing_type = 1
            break
        # CN(3)
        if str(nodei[-1]) == 'N':
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'TRIPLE':
                        if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                            electron_withdrawing_type = 1
                            break
                        elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                            electron_withdrawing_type = 1
                            break
        if electron_withdrawing_type:
            break
        # NO2
        if str(nodei[-1]) == 'N':
            o_one_check = False
            o_two_check = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_one_check = True
                        elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_one_check = True
                    elif edgej_bond_type == 'DOUBLE':
                        if edgej['head'] == index_i and str(
                                target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_two_check = True
                        elif edgej['tail'] == index_i and str(
                                target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_two_check = True
                    if o_one_check and o_two_check:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # COR, CHO
        if str(nodei[-1]) == 'C':
            rh_check = False
            o_two_check = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'H':
                                rh_check = True
                            elif str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['tail'])]
                                # check alkyl
                                rh_check = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                             invert_edge_type_dict)
                        elif edgej['tail'] == index_i:
                            if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'H':
                                rh_check = True
                            elif str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['head'])]
                                # check alkyl
                                rh_check = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                             invert_edge_type_dict)
                    elif edgej_bond_type == 'DOUBLE':
                        if edgej['head'] == index_i and str(
                                target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_two_check = True
                        elif edgej['tail'] == index_i and str(
                                target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_two_check = True
                        else:
                            break
                    if rh_check and o_two_check:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # CO2R, CO2H
        if str(nodei[-1]) == 'C':
            rh_check = False
            o_two_check = False
            o_one_check = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            if str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'H':
                                rh_check = True
                            elif str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['tail'])]
                                # check alkyl
                                rh_check = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                             invert_edge_type_dict)
                            elif str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                                o_one_check = True
                        elif edgej['tail'] == index_i:
                            if str(target_cata_node[int(edgej['head']) - 1][-1]) == 'H':
                                rh_check = True
                            elif str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                                check_list = [index_i, int(edgej['head'])]
                                # check alkyl
                                rh_check = utils.check_alhyl(check_list.copy(), target_cata_edge, target_cata_node,
                                                             invert_edge_type_dict)
                            elif str(target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                                o_one_check = True
                    elif edgej_bond_type == 'DOUBLE':
                        if edgej['head'] == index_i and str(
                                target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_two_check = True
                        elif edgej['tail'] == index_i and str(
                                target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_two_check = True
                        else:
                            break
                    if rh_check and o_two_check and o_one_check:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # CONH2
        if str(nodei[-1]) == 'C':
            o_two_check = False
            nh2_check = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'N':
                            check_list = [index_i, int(edgej['tail'])]
                            nh2_check = utils.check_nh3(check_list.copy(), target_cata_edge, target_cata_node,
                                                        invert_edge_type_dict)
                        elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'N':
                            check_list = [index_i, int(edgej['head'])]
                            nh2_check = utils.check_nh3(check_list.copy(), target_cata_edge, target_cata_node,
                                                        invert_edge_type_dict)
                    elif edgej_bond_type == 'DOUBLE':
                        if edgej['head'] == index_i and str(
                                target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_two_check = True
                        elif edgej['tail'] == index_i and str(
                                target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_two_check = True
                        else:
                            break
                    if nh2_check and o_two_check:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # SO3H
        if str(nodei[-1]) == 'S':
            o_two_check = 0
            oh_check = False
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            check_list = [index_i, int(edgej['tail'])]
                            oh_check = utils.check_oh(check_list.copy(), target_cata_edge, target_cata_node,
                                                      invert_edge_type_dict)
                        elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            check_list = [index_i, int(edgej['head'])]
                            oh_check = utils.check_oh(check_list.copy(), target_cata_edge, target_cata_node,
                                                      invert_edge_type_dict)
                    elif edgej_bond_type == 'DOUBLE':
                        if edgej['head'] == index_i and str(
                                target_cata_node[int(edgej['tail']) - 1][-1]) == 'O':
                            o_two_check += 1
                        elif edgej['tail'] == index_i and str(
                                target_cata_node[int(edgej['head']) - 1][-1]) == 'O':
                            o_two_check += 1
                        else:
                            break
                    else:
                        break
                    if oh_check and o_two_check == 2:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # CX3 X = F, Cl, Br
        if str(nodei[-1]) == 'C':
            x_check = 0
            for edgej in target_cata_edge:
                if edgej['head'] == index_i or edgej['tail'] == index_i:
                    edge_type_detail = int(edgej['type'])
                    edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                    if edgej_bond_type == 'SINGLE':
                        if edgej['head'] == index_i:
                            if str(target_cata_node[int(edgej['tail']) - 1][-1]) in ['F', 'Cl', 'Br']:
                                x_check += 1
                        elif edgej['tail'] == index_i:
                            if str(target_cata_node[int(edgej['head']) - 1][-1]) in ['F', 'Cl', 'Br']:
                                x_check += 1
                    else:
                        break
                    if x_check == 3:
                        electron_withdrawing_type = 1
                        break
        if electron_withdrawing_type:
            break
        # C6HnX(5-n) X = F, Cl, Br
        for edgej in target_cata_edge:
            if edgej['head'] == index_i or edgej['tail'] == index_i:
                edge_type_detail = int(edgej['type'])
                edgej_bond_type = utils.bond_type_detection(invert_edge_type_dict[str(edge_type_detail)])
                if edgej_bond_type == 'SINGLE':
                    if edgej['head'] == index_i and str(target_cata_node[int(edgej['tail']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['tail'])]
                    elif edgej['tail'] == index_i and str(target_cata_node[int(edgej['head']) - 1][-1]) == 'C':
                        check_list = [index_i, int(edgej['head'])]
                    else:
                        check_list = []
                    if check_list:
                        # check phenyl
                        phenyl_check, x_check = utils.check_phenyl_x(check_list.copy(), target_cata_edge,
                                                                     target_cata_node, invert_edge_type_dict)
                        if phenyl_check and x_check:
                            electron_withdrawing_type = 1
                            break
        if electron_withdrawing_type:
            break

    # proton donor
    # proton_donor_type: 0 -> none; 1 -> exists
    proton_donor_type = 0
    for itei in range(len(target_cata_node)):
        nodei = target_cata_node[itei]
        atom_name = str(nodei[-1])
        if atom_name.endswith('+'):
            atom_name = atom_name[:-1]
        elif atom_name.endswith('-'):
            atom_name = atom_name[:-1]
        if atom_name == 'H':
            proton_donor_type = 1
            break
    return double_metal_core, metal_core_type, electron_donating_type, electron_withdrawing_type, proton_donor_type

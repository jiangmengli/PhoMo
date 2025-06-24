# encoding=utf-8
import csv
import math

import utils
from config import *


def read_experimentfile_cataperform(experiment_file, molecule_list, edge_list, invert_edge_type_dict):
    print('Performing \'read_experimentfile_cataperform\' ...')
    # Read csv
    experiment_list = []
    abv_experiment_list = []

    experiment_list_temp = []
    with open(experiment_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        experiment_header = next(csv_reader)
        for row in csv_reader:
            experiment_list_temp.append(row)
    for row in experiment_list_temp:
        ex_dict = {}
        for i in range(len(experiment_header)):
            ex_dict[experiment_header[i]] = str(row[i])
        if str(ex_dict['Photosensitizer']) in list(experiment_notation_dict['Photosensitizer'].keys()):
            photosensitizer_id = experiment_notation_dict['Photosensitizer'][str(ex_dict['Photosensitizer'])]
        else:
            photosensitizer_id = len(list(experiment_notation_dict['Photosensitizer'].keys()))
            experiment_notation_dict['Photosensitizer'][str(ex_dict['Photosensitizer'])] = int(photosensitizer_id)
        if str(ex_dict['Sacrificial Donors']) in list(experiment_notation_dict['Sacrificial Donors'].keys()):
            sacrificial_donors_id = experiment_notation_dict['Sacrificial Donors'][str(ex_dict['Sacrificial Donors'])]
        else:
            sacrificial_donors_id = len(list(experiment_notation_dict['Sacrificial Donors'].keys()))
            experiment_notation_dict['Sacrificial Donors'][str(ex_dict['Sacrificial Donors'])] = int(sacrificial_donors_id)
        if str(ex_dict['Solvent']) in list(experiment_notation_dict['Solvent'].keys()):
            solvent_id = experiment_notation_dict['Solvent'][str(ex_dict['Solvent'])]
        else:
            solvent_id = len(list(experiment_notation_dict['Solvent'].keys()))
            experiment_notation_dict['Solvent'][str(ex_dict['Solvent'])] = int(solvent_id)
        if str(ex_dict['Gas']) in list(experiment_notation_dict['Gas'].keys()):
            gas_id = experiment_notation_dict['Gas'][str(ex_dict['Gas'])]
        else:
            gas_id = len(list(experiment_notation_dict['Gas'].keys()))
            experiment_notation_dict['Gas'][str(ex_dict['Gas'])] = int(gas_id)
        if str(ex_dict['Lambda']) in list(experiment_notation_dict['Lambda'].keys()):
            lambda_id = experiment_notation_dict['Lambda'][str(ex_dict['Lambda'])]
        else:
            lambda_id = len(list(experiment_notation_dict['Lambda'].keys()))
            experiment_notation_dict['Lambda'][str(ex_dict['Lambda'])] = int(lambda_id)
        try:
            add_abv_list = [int(ex_dict['Catalyst'])]
        except:
            continue

        # Catalyst graph analysis
        target_cata_node = molecule_list[int(ex_dict['Catalyst']) - 1]
        target_cata_edge = edge_list[int(ex_dict['Catalyst']) - 1]

        # Get graph scan
        double_metal_core, metal_core_type, electron_donating_type, electron_withdrawing_type, proton_donor_type = graph_scan(
            target_cata_node, target_cata_edge, invert_edge_type_dict)

        ex_dict['Double Metal Core'] = int(double_metal_core)
        ex_dict['Metal Core Type'] = int(metal_core_type)
        ex_dict['Electron Donating Group'] = int(electron_donating_type)
        ex_dict['Electron Withdrawing Group'] = int(electron_withdrawing_type)
        ex_dict['Proton Donor'] = int(proton_donor_type)
        add_abv_list.append(int(double_metal_core))
        add_abv_list.append(int(metal_core_type))
        add_abv_list.append(int(electron_donating_type))
        add_abv_list.append(int(electron_withdrawing_type))
        add_abv_list.append(int(proton_donor_type))
        # Reaction conditions
        add_abv_list.append(float(ex_dict['Catalyst Concentration_uM']))
        add_abv_list.append(int(ex_dict['Charge']))
        add_abv_list.append(photosensitizer_id)
        try:
            pcmm = float(ex_dict['Photosensitizer Concentration_mM'])
        except:
            pcmm = 0.0
        add_abv_list.append(pcmm)
        add_abv_list.append(sacrificial_donors_id)
        try:
            sdcmm = float(ex_dict['Sacrificial Donors Concentration_mM'])
        except:
            sdcmm = 0.0
        add_abv_list.append(sdcmm)
        add_abv_list.append(solvent_id)
        try:
            th = float(ex_dict['Time_h'])
            if not th:
                th = timecomp
        except:
            th = timecomp
        add_abv_list.append(th)
        add_abv_list.append(gas_id)
        add_abv_list.append(lambda_id)
        # Products
        try:
            co_product = float(ex_dict['CO'])
        except:
            co_product = 0.0
        try:
            if th:
                co_product_per = co_product / th
            else:
                co_product_per = 0.0
        except:
            co_product_per = 0.0
        add_abv_list.append(co_product)
        add_abv_list.append(co_product_per)
        try:
            ch4_product = float(ex_dict['CH4'])
        except:
            ch4_product = 0.0
        try:
            if th:
                ch4_product_per = ch4_product / th
            else:
                ch4_product_per = 0.0
        except:
            ch4_product_per = 0.0
        add_abv_list.append(ch4_product)
        add_abv_list.append(ch4_product_per)
        try:
            h2_product = float(ex_dict['H2'])
        except:
            h2_product = 0.0
        try:
            if th:
                h2_product_per = h2_product / th
            else:
                h2_product_per = 0.0
        except:
            h2_product_per = 0.0
        add_abv_list.append(h2_product)
        add_abv_list.append(h2_product_per)
        try:
            hcooh_product = float(ex_dict['HCOOH'])
        except:
            hcooh_product = 0.0
        try:
            if th:
                hcooh_product_per = hcooh_product / th
            else:
                hcooh_product_per = 0.0
        except:
            hcooh_product_per = 0.0
        add_abv_list.append(hcooh_product)
        add_abv_list.append(hcooh_product_per)
        try:
            co2_consump = co_product + ch4_product + hcooh_product
        except:
            co2_consump = 0.0
        try:
            if th:
                co2_consump_per = co2_consump / th
            else:
                co2_consump_per = 0.0
        except:
            co2_consump_per = 0.0
        add_abv_list.append(co2_consump)
        add_abv_list.append(co2_consump_per)
        ex_dict['CO2'] = str(co2_consump)
        ex_dict['Evaluation'] = int(ex_dict['Evaluation'])
        add_abv_list.append(int(ex_dict['Evaluation']))
        if co_product:
            ex_dict['CO_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['CO_Exist'] = 0
            add_abv_list.append(0)
        if ch4_product:
            ex_dict['CH4_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['CH4_Exist'] = 0
            add_abv_list.append(0)
        if h2_product:
            ex_dict['H2_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['H2_Exist'] = 0
            add_abv_list.append(0)
        if hcooh_product:
            ex_dict['HCOOH_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['HCOOH_Exist'] = 0
            add_abv_list.append(0)
        # Save
        experiment_list.append(ex_dict)
        abv_experiment_list.append(add_abv_list)
    return experiment_list, abv_experiment_list, experiment_key_list, experiment_notation_dict, experiment_post_descrip


def read_bad_experimentfile_cataperform(molecule_list, edge_list, invert_edge_type_dict, len_molecule_list):
    print('Performing \'read_bad_experimentfile_cataperform\' ...')
    # Read csv
    experiment_list = []
    abv_experiment_list = []
    for ele_i in range(len(molecule_list)):
        try:
            add_abv_list = [ele_i + len_molecule_list]
        except:
            continue
        # Catalyst graph analysis
        target_cata_node = molecule_list[ele_i]
        target_cata_edge = edge_list[ele_i]
        # Set ex_dict
        ex_dict = bad_experiment_dict
        ex_dict['Catalyst'] = str(ele_i + len_molecule_list)

        # Get graph scan
        double_metal_core, metal_core_type, electron_donating_type, electron_withdrawing_type, proton_donor_type = graph_scan(
            target_cata_node, target_cata_edge, invert_edge_type_dict)

        ex_dict['Double Metal Core'] = int(double_metal_core)
        ex_dict['Metal Core Type'] = int(metal_core_type)
        ex_dict['Electron Donating Group'] = int(electron_donating_type)
        ex_dict['Electron Withdrawing Group'] = int(electron_withdrawing_type)
        ex_dict['Proton Donor'] = int(proton_donor_type)
        add_abv_list.append(int(double_metal_core))
        add_abv_list.append(int(metal_core_type))
        add_abv_list.append(int(electron_donating_type))
        add_abv_list.append(int(electron_withdrawing_type))
        add_abv_list.append(int(proton_donor_type))
        # Reaction conditions
        try:
            add_abv_list.append(float(ex_dict['Catalyst Concentration_uM']))
        except:
            add_abv_list.append(float(0))
        try:
            add_abv_list.append(int(ex_dict['Charge']))
        except:
            add_abv_list.append(int(0))
        add_abv_list.append(0)
        try:
            pcmm = float(ex_dict['Photosensitizer Concentration_mM'])
        except:
            pcmm = 0.0
        add_abv_list.append(pcmm)
        add_abv_list.append(0)
        try:
            sdcmm = float(ex_dict['Sacrificial Donors Concentration_mM'])
        except:
            sdcmm = 0.0
        add_abv_list.append(sdcmm)
        add_abv_list.append(0)
        try:
            th = float(ex_dict['Time_h'])
            if not th:
                th = 0
        except:
            th = 0
        add_abv_list.append(th)
        # Products
        try:
            co_product = float(ex_dict['CO'])
        except:
            co_product = 0.0
        try:
            if th:
                co_product_per = co_product / th
            else:
                co_product_per = 0.0
        except:
            co_product_per = 0.0
        add_abv_list.append(co_product)
        add_abv_list.append(co_product_per)
        try:
            ch4_product = float(ex_dict['CH4'])
        except:
            ch4_product = 0.0
        try:
            if th:
                ch4_product_per = ch4_product / th
            else:
                ch4_product_per = 0.0
        except:
            ch4_product_per = 0.0
        add_abv_list.append(ch4_product)
        add_abv_list.append(ch4_product_per)
        try:
            h2_product = float(ex_dict['H2'])
        except:
            h2_product = 0.0
        try:
            if th:
                h2_product_per = h2_product / th
            else:
                h2_product_per = 0.0
        except:
            h2_product_per = 0.0
        add_abv_list.append(h2_product)
        add_abv_list.append(h2_product_per)
        try:
            hcooh_product = float(ex_dict['HCOOH'])
        except:
            hcooh_product = 0.0
        try:
            if th:
                hcooh_product_per = hcooh_product / th
            else:
                hcooh_product_per = 0.0
        except:
            hcooh_product_per = 0.0
        add_abv_list.append(hcooh_product)
        add_abv_list.append(hcooh_product_per)
        try:
            co2_consump = co_product + ch4_product + hcooh_product
        except:
            co2_consump = 0.0
        try:
            if th:
                co2_consump_per = co2_consump / th
            else:
                co2_consump_per = 0.0
        except:
            co2_consump_per = 0.0
        add_abv_list.append(co2_consump)
        add_abv_list.append(co2_consump_per)
        ex_dict['CO2'] = str(co2_consump)
        ex_dict['Evaluation'] = int(ex_dict['Evaluation'])
        add_abv_list.append(int(ex_dict['Evaluation']))
        if co_product:
            ex_dict['CO_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['CO_Exist'] = 0
            add_abv_list.append(0)
        if ch4_product:
            ex_dict['CH4_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['CH4_Exist'] = 0
            add_abv_list.append(0)
        if h2_product:
            ex_dict['H2_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['H2_Exist'] = 0
            add_abv_list.append(0)
        if hcooh_product:
            ex_dict['HCOOH_Exist'] = 1
            add_abv_list.append(1)
        else:
            ex_dict['HCOOH_Exist'] = 0
            add_abv_list.append(0)
        # Save
        experiment_list.append(ex_dict)
        abv_experiment_list.append(add_abv_list)
    return experiment_list, abv_experiment_list


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

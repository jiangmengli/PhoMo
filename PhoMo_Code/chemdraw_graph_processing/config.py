# encoding=utf-8
import math


# Dataset version
config_chemdraw_record_version = 'PhoCata'
config_chemdraw_causal_version = 'CauReaNode'
config_chemdraw_pred_version = 'PredPhoCata'


# Time completion hyperparameter
timecomp = 8


# edge numbering from zero
edge_numbering_from_zero = True


# Atom attributes
atom_attributes = {
    'Zr': {
        'possible_atomic_num': 40,
        'possible_degree': 4,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'Br': {
        'possible_atomic_num': 35,
        'possible_degree': 1,
	    'possible_numH': 1,
        'possible_number_radical_e': 0
    },
    'C': {
        'possible_atomic_num': 6,
        'possible_degree': 4,
	    'possible_numH': 4,
        'possible_number_radical_e': 0
    },
    'N': {
        'possible_atomic_num': 7,
        'possible_degree': 4,
	    'possible_numH': 3,
        'possible_number_radical_e': 0
    },
    'Cl': {
        'possible_atomic_num': 17,
        'possible_degree': 1,
	    'possible_numH': 1,
        'possible_number_radical_e': 0
    },
    'P': {
        'possible_atomic_num': 15,
        'possible_degree': 4,
	    'possible_numH': 3,
        'possible_number_radical_e': 0
    },
    'S': {
        'possible_atomic_num': 16,
        'possible_degree': 4,
	    'possible_numH': 2,
        'possible_number_radical_e': 0
    },
    'Si': {
        'possible_atomic_num': 14,
        'possible_degree': 4,
	    'possible_numH': 4,
        'possible_number_radical_e': 0
    },
    'H': {
        'possible_atomic_num': 1,
        'possible_degree': 1,
	    'possible_numH': 1,
        'possible_number_radical_e': 0
    },
    'F': {
        'possible_atomic_num': 9,
        'possible_degree': 1,
	    'possible_numH': 1,
        'possible_number_radical_e': 0
    },
    'Mn': {
        'possible_atomic_num': 25,
        'possible_degree': 6,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'Fe': {
        'possible_atomic_num': 26,
        'possible_degree': 8,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'Ni': {
        'possible_atomic_num': 28,
        'possible_degree': 6,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'O': {
        'possible_atomic_num': 8,
        'possible_degree': 3,
	    'possible_numH': 2,
        'possible_number_radical_e': 0
    },
    'Co': {
        'possible_atomic_num': 27,
        'possible_degree': 6,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'N+': {
        'possible_atomic_num': 7,
        'possible_degree': 4,
	    'possible_numH': 3,
        'possible_number_radical_e': 0
    },
    'Zn': {
        'possible_atomic_num': 30,
        'possible_degree': 8,
	    'possible_numH': 0,
        'possible_number_radical_e': 0
    },
    'Cu': {
        'possible_atomic_num': 29,
        'possible_degree': 8,
        'possible_numH': 0,
        'possible_number_radical_e': 0
    }
}


normal_metal_atom_list = ['Fe', 'Zn', 'Co', 'Ni', 'Mn', 'Cu']


noble_metal_atom_list = ['Ir', 'Ru', 'Re', 'Os', 'Pd']


cos_check_list = [1]
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
    if i == 90:
        cos_check_list.append(0)
    else:
        cos_check_list.append(math.cos(math.radians(i)))
cos_check_list.append(-1)

stereo_replacement_dict = {
    'SP': ['Mt', 'Sg', 'Lr', 'Gd'],
    'SP2': ['Hs', 'Db', 'No', 'Tm', 'Ac'],
    'SP3': ['Bh', 'Rf', 'Md', 'Fm', 'P', 'Si', 'Pa', 'U', 'Co', 'Ni', 'Pu', 'Er', 'Yb', 'Cuq'],
    'SP3D': ['Tb', 'Th', 'Es', 'Cm', 'Ho', 'Np', 'Cup'],
    'SP3D2': ['Mn', 'Cf', 'Am', 'Dy', 'Zn', 'Cuh'],
}

experiment_notation_dict = {
    'Photosensitizer': {
        '-': 0
    },
    'Solution Additives_M': {
        '-': 0
    },
    'Sacrificial Donors': {
        '-': 0
    },
    'Solvent': {
        '-': 0
    },
    'Gas': {
        '-': 0
    },
    'Lambda': {
        '-': 0
    }
}

bad_experiment_dict = {'Catalyst': '', 'Double Metal Core': '', 'Metal Core Type': '', 'Electron Donating Group': '',
                       'Electron Withdrawing Group': '', 'Proton Donor': '', 'Catalyst Concentration_uM': '-',
                       'Charge': '-', 'Photosensitizer': '-', 'Photosensitizer Concentration_mM': '-', 'Solution Additives_M': '-',
                       'Sacrificial Donors': '-', 'Sacrificial Donors Concentration_mM': '-', 'Solvent': '-', 'Time_h': '-',
                       'Gas': '-', 'Lambda': '-', 'CO': '-', 'CH4': '-', 'H2': '-', 'HCOOH': '-', 'CO2': '',
                       'Evaluation': 0, 'CO_Exist': 0, 'CH4_Exist': 0, 'H2_Exist': 0, 'HCOOH_Exist': 0}

experiment_key_list = ['Catalyst', 'Double Metal Core', 'Metal Core Type', 'Electron Donating Group', 'Electron Withdrawing Group',
                       'Proton Donor', 'Catalyst Concentration_uM', 'Charge', 'Photosensitizer', 'Photosensitizer Concentration_mM', 'Solution Additives_M',
                       'Sacrificial Donors', 'Sacrificial Donors Concentration_mM', 'Solvent', 'Time_h', 'Gas', 'Lambda', 'CO', 'CO_Per',
                       'CH4', 'CH4_Per', 'H2', 'H2_Per', 'HCOOH', 'HCOOH_Per', 'CO2', 'CO2_Per', 'Evaluation', 'CO_Exist',
                       'CH4_Exist', 'H2_Exist', 'HCOOH_Exist']

experiment_post_descrip = 'Data: Catalyst, Double Metal Core, Metal Core Type, Electron Donating Group, Electron Withdrawing Group, Proton Donor, ' \
                          'Catalyst Concentration_uM, Charge, Photosensitizer, Photosensitizer Concentration_mM, Solution Additives_M, Sacrificial Donors, ' \
                          'Sacrificial Donors Concentration_mM, Solvent, Time_h, Gas, Lambda\nLabels to regress: CO, CO_Per, CH4, CH4_Per, H2, H2_Per, HCOOH, HCOOH_Per, CO2, CO2_Per, \nLabels to classify: Evaluation ' \
                          '(where 0 -> Not catalyst, 1-> Bad, 2 -> medium, 3 -> Good), CO_Exist, CH4_Exist, H2_Exist, HCOOH_Exist'

pred_molecule_key_list = ['Catalyst', 'Double Metal Core', 'Metal Core Type', 'Electron Donating Group', 'Electron Withdrawing Group',
                            'Proton Donor']

pred_molecule_post_descrip = 'Data: Catalyst, Double Metal Core, Metal Core Type, Electron Donating Group, Electron Withdrawing Group, Proton Donor, '

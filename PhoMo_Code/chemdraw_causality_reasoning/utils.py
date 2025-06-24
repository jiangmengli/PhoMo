# encoding=utf-8
from config import *


def read_chemdrawfile(mol_ct_file):
    # Read text
    lines = []
    with open(mol_ct_file, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            lines.append(line)
            if not line:
                break
                pass
            pass
        pass
    file_to_read.close()
    return lines


def write_chemdrawfile(mol_ct_file, mol_ct):
    fp2 = open(mol_ct_file, 'w')
    for ele in mol_ct:
        fp2.write(ele)
    fp2.close()
    return True

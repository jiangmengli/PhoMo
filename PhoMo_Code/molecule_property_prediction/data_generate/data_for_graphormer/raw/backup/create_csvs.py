# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import pandas as pd
from ogb.utils import smiles2graph
import torch
from pysmiles import read_smiles
from rdkit import Chem
import networkx as nx
import sys







if __name__ == '__main__':

    for MOL_NAME in ['molbbbp','moltox21','molesol']:

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/edge.csv") as f:
            reader = csv.reader(f)
            edge_ori =[row for row in reader]
            # print(rows)

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/edge-feat.csv") as f:
            reader = csv.reader(f)
            edge_feat_ori = [row for row in reader]

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/graph-label.csv") as f:
            reader = csv.reader(f)
            graph_label_ori = [row for row in reader]

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/node-feat.csv") as f:
            reader = csv.reader(f)
            node_feat_ori = [row for row in reader]

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/num-edge-list.csv") as f:
            reader = csv.reader(f)
            num_edge_list_ori = [row for row in reader]

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/num-node-list.csv") as f:
            reader = csv.reader(f)
            num_node_list_ori = [row for row in reader]

        with open("/home/gaohang/Researches/ChemGraph/dataset/" + MOL_NAME + "-su/test.csv") as f:
            reader = csv.reader(f)
            test_set_ori = [row for row in reader]

        print("=== files loaded ===")

        #'[H+].C1=CC=C(C=C1)C(=O)N[C@@H]([C@H](C2=CC=C(C=C2)OC)C(=O)N[C@@H](CCC(=O)O)C(=O)O)C(=O)O'
        smiles_list = [ '[H+].[H+].[H+].CC(=O)N(CCCCCNC(=O)CCC(=O)N(CCCCCNC(=O)CCC(=O)N(CCCCCN)O)O)O', 'CCCNC(C)=CC12C(c3cnc(C)n3C)=NN=C1C2CC=N', 'CNC(C)C=CCCC=COCNNCF', 'CCC=C(CC(Cl)C1=CCC(=N)CN1Cl)C(C)O', 'CC(CF)(C(F)Cl)N(CC(F)P[IH]CCC(C=CO)CF)NN', 'CC(C)NPC(O)=CC[NH2+]F', 'ClCCNC=CC1C=CNNC1', 'CCCCCN(C)C(Br)CF', 'CCC(P)=C(C)CN(F)Br', 'CC(C[N+](=N)c1nc(=O)ncc(=O)s1)C(=O)NI', 'O=C(C=CCC[IH]I)CCN(O)Cl', 'CC1=Cc2ccc(cc2CN=O)C(CCN=CF)CN1NBr', 'C=NCC1CC(Cl)C=NN(C)N(CNO)C2=CN=C([SH]=C2)C(S)C1', 'COCSPNC(C)=CCNCCF', 'CCCC1=C(Cl)NC=C(C=C(C)CI)N1', 'NC(CCCCCCCCCNC=O)C1CN(Cl)C1F', 'N=C1C(CC2=CC=CC2(O)CNC(F)NF)CN1S', 'CC=CCC(F)NCC=CSNCC1(I)C(Cl)N=S1(=O)CCC=NF', 'C=CCN[N+](=N)CNN(I)N1CNC(N=NCF)=N1', 'CP(P)CN=NSN', 'C[IH]CC(=CF)[PH](N)(C=C(O)CF)CCC(CCCl)=C1C=C1', 'CCNCCC(=O)F', 'CCC=COC(F)C(F)NCC(=O)OCNOC', 'COC1OC(=CC(C)CF)C=C1O', 'CCCC(F)C=CF', 'C=C(O)C(O)[O+](CCNC(=O)C=O)[IH](C)(C)I', 'CCC(=NC=C(F)CF)OC(C)NCCC=O', 'CC(C=CC(=O)O)CN', 'CC(CI)CC(=O)C(CN)OC(N)=O', 'CCOC=C(C)C(F)CCNC1CO1', 'CC=CCOCC(CN)CN', 'C=C1NCC[O+](CCOCF)C(CC)C(F)CSCC1O', 'CCCC=CC(=O)C(C)(O)C(N)O', 'CC(OF)C(O)F', 'CCCCOC1C(OCOCC)NC(NI)C1C', 'CC=C1C=C(NCC(F)F)CC(C)C(=N)C=CCCC(=[IH])CCC(CC)C1', 'CC=COC(C=O)C(=CF)N(C)C', 'CCCCCCCC=CCC=CCC(F)NCC', 'CCOC(F)C=COC(C)C', 'CC1=C(CC=NC(=O)NCCC(C)C)C(C)CO1', 'C=C1OCC=CN=C(O)CC(C)C(=O)CCC2=C=CC1=NC=C2CC', 'CCN=CC1CCC(=O)CCC(=S)SCC1(C)[IH][IH]Cc1ccnnc1', 'CC=CC(=C(I)I([OH+]O)C(C)CC)C(S)I', 'CC=C(C)N(C)C(C)C=NCC(C)S', 'CCNCNN(C=CC(=O)CC)C1NC1OC(C)(F)NOC1=CCCC=C1', 'CCC1=PC2=CC=CC[IH]CC1C2', 'CC(=C[IH]I)C(F)CCCC(C)C(=[IH])C(F)CN=O', 'CC=C1[IH]CNC2(P)NC(NOCCC)CC13N=NNC3c1ncccc12', 'CCCCCC=CC1C=NN=C(C)CC=CC=CCC=C1I', 'N=NC1C(=CCC=NF)CC1=Cc1cc1=NPPCl', 'CCN1C=CI=[O+]CN1', 'CCCCCNC(C)OI(I)[IH]C1=NI=[O+]C1=CC(C)=CCCF', 'CCCCCC(I)CCC(=O)NC(=O)NSCCC', 'CCOCNCCCCC=C1CC(Cl)=CC2=CC1=CC=[PH]2O', 'CC1[IH][IH]#CC(C)(C)C1C(C)(P)NC(=O)N=CC(=P)Cl', 'C=CC=C(I)CC(P)C(C=NN=NC=CSCCC)CC(=O)CCN', 'CCC=CCCC=CP=CCNC(Cl)OOC', 'CCN1CCC=C(C)C1(C)C(=CI)C(O)OC(C)NC', 'CC1=C(I)C(N[N+](=[IH])NC=CCC[O+](C)O)=CC1', 'CC=CC=CCC', 'CCCC(C)C(C)=O', 'CC=CC(=O)CCF', 'CC(O)OCC(=CCCN(O)OC=O)OCl', 'C=NC(C)OCOCC1=CN=CCO1', 'N#CCOC(C=NN)COCC[OH+][OH+]C(=[OH+])N(NN)OO', 'CC1=CCC(CO)C(CCF)C(=O)OCCCCC(O)C1=N', 'CN[O+](O)CONCC(N)C1C=C(N=CCOC(N)=O)COC1', 'OCCNC1OCN[O+]1O', 'CCOC(=O)C(=O)N=C(NCCC(=O)OCOC=CCNCl)ONF', 'CN1CC2CCN(CO2)C2(C=O)OCOC2C1', 'CCOC(=O)OC(OC=CCC1COCNCO1)C(O)C=CCN', 'CCON(C)C(CO)O[IH]OCOCCOCNC', 'CC=COCNCNS(=O)(=O)OCBr', 'CC1=CCC(C)[IH](C)=C1CNCC=COCON[O+](C)OCN', 'CC[IH]CCC=CC(=O)OCOO[O+](Br)OC', 'NCOC=COOC(=O)OCCCC12CCC1C2', 'NCC(S)OCCO', 'CN1OC2(S)C=CC1CCNC(=O)CC(C1=NC(C)(O)O1)C2', 'OC(CCOCSNCNCN=[IH])OCN1NCOC=[O+]1', 'CN(C=CCCBr)C(=O)COCCOCN', 'CC1CCCCCC1O', 'C(C(=O)C(C(=O)O)N)C(C(C)C)C(=O)O', 'COC1=C(C=CC(=C1)/C=C/C2=CC=[N+](C=C2)CCC(=O)O)C(=O)O', '[H+].[H+].[H+].CC(=O)N(CCCCCNC(=O)CCC(=O)N(CCCCCNC(=O)CCC(=O)N(CCCCCN)O)O)O', 'O=C(C=C1NCCOC=C1CF)CCF', 'CC(CC(F)Cl)C(C)C(Cl)C1NCON2CC1=CS2', 'C[IH]NCN=C(C)S(=P)(=P)NC(F)N1NCCI1C', 'COCC(F)C=CCCOCC=O', 'CC(CC=CF)NCCN=NC=CNOCNNC=C1C=CCCC1', 'CC1C2=CC=CCC23CN(C)N(C(F)NCCN1F)N(C)CC3CCl', 'FCNC(CP)NC1=CCCC1S', 'CN(NC(O)C(N)CCC=NCF)c1cccnc1CC1CN1', 'C=C(C1=C2C[IH]C21C)[NH+](Cl)C=C(N=C[IH][IH]C(C)N)P1CC1', 'N=C1CC=C(C(=S)I)C(O)=N[IH]CCC1', 'C=CN=C(C)OC(C)CC=CC1=Cc2cc1n(CCl)[n+]2I', 'CC(CCCF)CC(=O)[IH]C(=O)NCCN[IH](I)=CC=CSBr', 'CC(F)C1PP2NC1N(I)C=[N+]2[OH+]C1(O)NOC1O', 'CNOC(=O)N(O)P(N(N)NC(C)I)P(I)NCC(=O)C(=S)N(F)NF', 'COC(CC=CI)(Nc1cccnc1)C(C)C', 'COCC1NC1C1=CC=CC(C)N1SC', 'NCSCCC1CCNC1', 'COc1ccc(C)cc1CC(=O)C1CC(CN2CCCCC2)SN1P', 'C=C(CNN(Cl)NCF)C1=CCNNC1', 'CCOC(=CC(=O)CNCCN)CNO', 'CC(N)[IH]COOCN(C)C', 'CC1C23C4C25C2C6(C)C7C89CC6%10C6C%11CCC%12C%13CC(C)(C%14CC%13%14)C%13%14C%15C(C%11%16C%12C2%16C13C%158C6%10)C%13(C)C9%14C745', 'CCC(N)=CC(C)=NCNCC[IH]CCCCCCCC=O', 'CCO[IH]CC=CCC1CCN1C1CC(C)OCCC2CC21', 'C=CNCCCC(C)=O', 'CC(C)NN=CCC(F)CN=CC(=CC=CC(O)O)CI', 'COC=CC=NCCOCOC(C)N', 'CCNCNC1C2=CC(=O)CC1C(C=O)CCCCC2', 'CCCNC1CC(C)N2CC1C(Cl)c1cccc2n1', 'COC=CCCC1=CNC=CN=C1CF', 'C=CC(F)CCCCCC(C=CC=COC)C(C)I', 'CCCNC=CC(C)C1=C(CC=CC2=NCCC(C)C2)CC=N1', 'C=CN(CC1CCC(N)CCCC1CNOC)C(O)F', 'CCCCNC(=O)CCC(=O)CCCC(C)CCC', 'CCNCOC(CCCC=NC(=O)C=CNC)C1NC1N(C)CN=CBr', 'CCC=CCC(=O)C[O+]=CCC[N+](=O)CC1CC=CCC1=O', 'CC1CCN(C)CC=C1C1=CC=CCN(C(C)F)N1', 'CC1=CC=C(C)I(CCCCC2=CC=CCNOCC=N2)C(=O)CCC(O)C1', 'CCNC1=CNC(CF)CCN=C2C=C2N(F)CN1', 'CCCCC(C)=C1OC2NC(=O)C(=C2N)ON1SCC(C)=O', 'CC(=O)CC=C(CC=CCI)NC=NC1=CC(C(C)(F)OCN)C[IH]1', 'C=CCCCC(=O)NCCC1(O)[IH]CCCC1(C)C[IH]P', 'NNNCC[IH]CCCCCCC(N)=O', 'COCC(C)PP(P)[PH](N)=P', 'CC(CO)=NC=CC(=O)PC(P)C1=CC(=O)I=C(O)CCCC1', 'CCC1=CC=CC(=O)C(F)NCC2=P(C)(CC=C2)CCN(I)CC1', 'COC=CONC(O)CCCC=CC(C)[PH]1=CC=CC(C)C(=P)CI(C)S1', 'CC(CCCF)CCCI', 'ClCNC1CCCN(I)CCC2C3PCCN2C3N1', 'CS[IH]C(C(=O)OC(O)O)P(C)CC=CCC=CN', 'CCOCCNCCCC=CCP1CPPCC1C(O)P', 'O=CCCCC=S1CCC=CC12C=CCCC2', 'CCOC=C(CC=CCCCCCCCCI)[IH]OC(=O)S', 'CCC(=CC=CC(C)CC=CC=CC(P)C(=O)PBr)CF', 'CCC(F)CC=CCCCNCC(P)C1=NC1I(I)N[N+](=O)P', 'CCNC(=O)NCC(C)COC(F)C1=PCC(C)=C1', 'CCC(C)=CCPCC(C)CN', 'CC1CN(CN)CN[PH](P)(P(N)P)PN1', 'CCN=C1N(CN)[N+]12CC=CC=P2', 'CC(CCCCC(C)C(=O)CCO)=C(O)C(N=S)NCBr', 'CCCCNC=CCC(O)=NC(CCC=C=O)=C(O)O', 'CC(N[IH]N=COC(C)I)=[O+]C(=O)F', 'CN=CC=CC(=O)C=CC=NC=[PH](CN)OCN', 'CC=CC=C(C)CO', 'CC(=N)C1OCC(=O)C=C(C)OC1CCN', 'CC(C)CCC=NCC(C)F', 'C[IH][IH]COC(I)=[O+]C1=CC(=O)OC(C)C=NC(C)=C1', 'CC(=C(F)CC(CF)OO)C(C)CC1C=CCCC1', 'CCCC1=CCC(CO)=NC=C1C=C1CCCN=CC(C[PH](=O)O)C1', 'CCN=CC1=NC=CCC=C1CC', 'CC1=NC=C(C(=O)F)[O+]2CCC(C)C(N)=C2N1', 'CC1=CC2C3(C(C)C)C(OC=O)CCC3(C)C23C(=CCCN3C)C1', 'FNOCN=CCCNCC(OCI)OCI', 'CC1CNC(CNOCS)C1C[O+]=CC[N+](=O)O', 'CO[O+](O)CN(O)C(C)CCCNCC(O)CCCOCCCC(C)F', 'C=C(N)C1=CCNCC=C1O', 'COOCC(F)CC(CCF)CCNCC=O', 'CC1C=CNC(CCCOC2CCC2CCF)C1(C)C', 'C=C(CCC(N)(F)NC)CC1C=CCN1CCC(NC)(OC)OC', 'CCC[IH]CCC=CCCCCCCOCNO']

        # 'COC1=CPCN(N)NC12PCNCC2(C)CC1CC=C(C)C(OC)CC1'
        smiles_list21a = ['CC(C)COCN(C)OC(=O)C=CC=CCCC1C(O)=CC[N+]1=O', 'CCC1=N[SH]=C[N+]2=C1CC1C=CC=Cc3cccc(n3)CCCCC2=N1', 'CCC1=CC(C)C(C(C)C=CCNC(C)SC)CCS12(=O)CC2CC', 'CC1C2=CC=CN1CCC([NH+](C)O)=CC=CCCN1C2CNCN1C', 'NCN1NCC(Cl)CNCC2=C(CCC2)C(Cl)C=CCNN1', 'CCC1C=CNC(=O)CCC2=C[IH]CC1(CC)N2C', 'COCCCCCCC1N=C(C)CSCCC1C', 'CCC(CC(=NC=CNCCC=CCC#N)C(N)O)SC', 'CNC(CNC1(C)CC=C(C)C1C)N(N)O', 'CCC=CCC(CC)NCCC(=O)NCC(=O)F', 'CC(C=C(N)CNN=C=O)OCC1NCCCCC1=O', 'CCC1CCCC(C)C(C)C1', 'CCC(=CCN(C)OC(C)=NCO)CC(=O)[O+](C)O', 'CCCCC(C)=CNC(=O)C=CC=CCCCCc1ccccc1', 'CC1SCCC(=O)NC(CC2(Cl)CCCC2)C1C', 'CC(CCC=CNC(C)C)CN(C)CCC1=CCCC=NN=CC=C1', 'CCC(CCCNC(C)N)CCC(=O)CC1=CC=CC1C', 'CCCCCC(C)NCCC=CC1C(C)N(CCl)C=CC12C=CC=C2', 'CC(C)CC(=O)OCCCC=NCC1NCOCC1C']
        smiles_list21b = ['CCCCC1CCCC1CC(O)NCC1=CCCC(=O)C=CC1OC', 'CC1=C2CC(=N1)CC2C=[N+]1CC(=CCS)CCC(CN)O1', 'CCC1C=CC(C2=CN=CCC2)CCC=[SH](=O)NC=[N+](O)CC1', 'CCNCC(=O)C=C(C)C1CC23CN(C)CCC1N2C3', 'CC(C=O)CC1CC=CC1(c1ccccc1SCl)S(=O)(=O)NC=C[N+](F)=CC=O', 'CCCCCCC=NCC(=O)C1CCOC(C)C1', 'CC=CCCC(=O)C=C(N)C=CCOC=COC(F)CNC(=O)Cl', 'CC(=O)C(NCCC1(N)NCCS1)C(C)C', 'CC(NCCC=C1CCC1C(C)C)NCCC(Cl)C=NCBr', 'CC1=NC2C(=CC(CF)CNN)CCC(NC(C)C)NN2C1', 'CCCSCCC(C)SCC1C=C(C)CNC1', 'CC=CCCOCC1=NC=CCCC1CC', 'O=CC1CC2CC3(CC(F)P)CC2=CC1C3C(=O)CC1C2CCCC(C=O)N21', 'CC=NCCCOCNCCN=CC(C)N1C=CC=CC1', 'CCC(=O)CC(C=CC(=O)CCC(=O)SC1=CC1C)=NCC(C)C', 'CCCC(C)NCN(C)CSCC1CC=CC=C1CC=O', 'CCC=CC(F)C1=CCNC(C)C1N(C)C(C)COCC(C)=NCC', 'CC(=O)CN(C)CCC=CC=C(C)N', 'CCC=C(F)C1=C(CC(C)(C)C=CC=CC=CBr)C(C)CCOC1', 'CCCCN=CCOCC=CC=CCNCC']
        smiles_list21 = smiles_list21a + smiles_list21b

        smiles_list22a = ['CCC(O)CC=COCOCOC(OC1OC1CC)OS(=O)OCCCF', 'C=[SH]OCC(C)O', 'COPC=C(P)OCCF', 'NNC(OOOC(O)=S)C(CF)C1CC=C(P)C1', 'CCOCCN(O)COSN(N)O', 'CC[SH]1(=O)OOCOC1OO', 'CCCOCP=PF', 'CN(CCOOCP(P)CN)SP(C)C', 'FC(F)OOCP=P', 'C=C(C)OC(C)=O', 'CS1(N)OCOOO1', 'OCCSOC1OCO[SH]1OS', 'COOO[SH](N)NO', 'COC[SH](OO)C(CO)COOO', 'BrSCCOOCC1OOOOSCNC12CO2', 'C=COC(OC)N(F)SOC(=O)O', 'C=COC(OOC)C(C)=C(F)CF', 'C=C(NOOC)SO', 'OCPNC1=CC=CC1F', 'CC1[SH]2C[SH](OCO)O[SH]1OC=CO2']
        smiles_list22b = ['COCOC=CCC(C)O', 'CC1CC(C)C(CO)C1', 'CC(=O)OCC=C(F)CCCCC(C)O', 'CCOC(F)=CCSOC(C)CCO', 'C=C(CC(C)CC1C(COC)CCC1OC)OC', 'CC(F)=CCCCCF', 'C=C1CCC(C=COC=O)C(C)OC(=COO)CO1', 'C=C(CCCC)OC(C)O', 'C=C(NCF)C(OC=C(C)OC)C(CO)=C1OC1F', 'CCC(C)CC1=C=CCCCC2=CCC(C)CC(C)(C2)C(C)=C1', 'CC(C)CCCC=O', 'OOC1=C(COCOF)C=CC1', 'C=PC(=C)OCCF', 'CC(CC(C)O)=PC(COCCCF)=C(COC(C)CF)OO', 'COC1C(C)C2=CCC1C(C)(O)OC(O)=CC=CO2', 'COCOC=CO', 'CCC=CC(C)(O)OOF', 'CC(C)C(=O)OF', 'C=C(CC)COC=CCCCF', 'CCC(=COOC)C(O)=COc1cccc(C)c1']
        smiles_list22c = ['CCOC=COC=CCOOS', 'COCCOP(P)N(P)C(C)N', 'C=C(OPCC(OF)OCOP)N1CCOP1P', 'O=C(C=CF)CPCOC1=CCNOCO1',
         'CCOC(CP=CC=CC=C(OP)N1C(O)OC1OO)=NCO', 'CC=NC(CN)O[PH](=P)c1ccccc1', 'C=C1C(COC(N)NO)=NPPN1C',
         'C=C(C)C(O)(OF)N(CNCO)OCCOO', 'FCC=CP[PH]1=CC=CCO1', 'CCOC(F)(F)C=CCF', 'CCNO[PH](P)=CN[SH](P)CPC(C)=O',
         'FC1=COCSNCPP1', 'C=CCCCOP(O)ON(CF)OOC', 'OOC(O)COCOCOS', 'CCOOC(=O)C(O)P', 'C=C(NC)OP=COCOC(C)F', 'COOCCOCPOCSOC',
         'FCC(F)CF', 'C=C(C)NPO[PH]1=PC(OOCN)=CC=CO1', 'CO[PH](=P)COP(O)P']
        smiles_list22d = ['FC(F)S(=S)CS', 'C=CC[PH](F)(F)CSOCCCF', 'CCOCCC(CF)PCCCC=O', 'CC(C=O)OCC=CC(F)CCC=O', 'CSSCCC=CCSCCSCCNSCOO', 'CSNOCC(=O)OCSOCOC(=O)Oc1ccc(C)o1', 'CC1NPCCOP=CO1', 'CCCCCC(C)[SH](S)CSO', 'CNC(NCC(C)OF)OC', 'CCC=CNC(S)NSC1CCCC1F', 'CCCO[SH](C)C(O)PSC=O', 'CSC(C1=C(F)COO1)[SH](C)P', 'CSPOCCOC(F)OPOS', 'CSP=PC(CC1OCOSOC1(C)C)PCP', 'C=C(C=CCF)C(F)CC=CC=O', 'O=C(O)C=CCF', 'COC=C(CF)OCCCCC(=O)C(C)C(O)CF', 'CC(N)C(C)COCCOC=CCCF', 'C=C(O)OC(=CCCC=O)CC(F)C(C)CC', 'C=C(C=C(CCOC=COC(=C)F)CN(C)O)CCCO']
        smiles_list22 = smiles_list22a + smiles_list22b + smiles_list22c + smiles_list22d

        smiles_list23a = ['CCC(O)(O)COCS(=O)OCC(C)F', 'COOC(PP)C(O)=CCF', 'CC(=CC(OF)OCOOOCO)PP', 'COOCC(C)CC[SH](O)O', 'COCS(C)(O)C(=CO)C(O)O', 'CN(O)ONOF', 'COC(C)NC(C)C(OC)OOCCF', 'COOCC=COC(OO)[SH](=O)(COCOCO)OOC', 'COOSCOC(=O)OCOCC(F)F', 'CC1O[O+]2[OH+][O+](CCOCOC[O+](S)CCO)OS[SH]12', 'CSN(SC[O+](N)C=C(C)CF)C(O)C=O', 'CCOC(=O)C(S)OOC(CS)OO', 'CCOC1(C)COCC[O+](O)OCS1', 'CP(P)O[NH+](CF)CN=CP=PN=P', 'CC(N)COCSN(C)C=CC(F)F', 'C=C(CF)OOSOC(=O)C(C=CCC)NCC', 'C=COCNOSC[O+](CF)C(O)=[OH+]', 'C=C[OH+][O+]1O[O+](S)OC1=C(C)OCOOSO', 'CO[SH]1(=O)C(C[O+](CCF)CCN(C)C(CO)CP)SO[O+]1S', 'C1ONS1']
        smiles_list23b = ['C=C(CCCOC)OC(C)C', 'CC(C)O[O+]=C[O+]=C[O+](C)CCC=CO', 'CCOOC(CC)C1COC(C)=CCO1', 'C=COC1CC(NC(C)O)C1CF', 'C=CC(O)N=CCSO', 'CC1C=CO1', 'COCC(C)C(CC(F)CCC=O)OC', 'C=C(C)OC(C)=C(OCC(C)C=CCCCF)C(CF)OC', 'C=CC(C)C(C)OC1CCCC1', 'C=C1C=COC(O)CCCC1C', 'CCCOC(OC)C(CCF)COC(=O)CF', 'CCCC(OCF)C(=O)O', 'C#CCC(C)CCC(=O)CCCC(C)CCC(=CCF)OO', 'COC1=[O+]CC(O)(CCCF)O1', 'CC(=CCCCF)OCC(F)CCC(C)O', 'C=CC(O)OC(C)CC[O+]=C', 'CC(=O)OCC(C)C(C)C', 'C=CC(F)CC(F)OCC(C)F', 'COCCCCC(C)OC=COCCN=C(O)O', 'CCC1OC=C(C(O)C(=N)O)CC1[O+]=O']
        smiles_list23c = ['COOS[O+]=[O+]OS(=O)[O+](C)P(C)OCCO', 'COCSN(OCSO)[OH+][OH+]P', 'C=C[OH+][OH+]OC=[O+]OCCOC=CPO[O+](O)Br',
         'O=P[O+](PC=CO)C(P)OC=PCCOC[O+](O)I', 'OC1C[O+](OS)S(C(=P)OCC=C[O+]=CCF)=[O+]1',
         'N=C1C[O+](OO)C(O[SH](OCP)[O+]=CC=CCOF)SO1', 'OCC(O)[OH+]NSCP(P)OOC(F)F', 'CCNSC(P)C(=P)C(C)=P',
         'CCC=CC=C[O+]1CO[O+](S)C1F', 'FCC=C(F)CC=[O+][PH](=P)CNC[O+](S)C=CC[IH]CF', 'C=C(OSCOCOOC(F)F)O[PH](=P)OC[O+](C)P',
         'C[O+](F)C=[O+]OC(CO[OH+]S)OSSOOP', 'OC(C=CF)C(O)OCOCC[O+]1PC=C(F)C=C1CF',
         'CC[O+](C)C(O)[O+](ON)O[PH](=O)S[O+](OC)OP[n+]1ccccc1F', 'CI=C1Cc2cc(ccp2)C2=NC1=CPC2',
         'OCOC(C=PF)NC(=CPOCCOPCOCCF)OP', 'O[O+]1CC2O[O+]([O+]=COC(OP)SP)C21',
         'CC1C(OCOC[O+](C(OS)OCO)[O+](C)CO)C=C2C=CCC21', 'COCO[O+](C[O+](P)CC=CO)C[O+](P)COP', 'C=C(F)C(F)OC1COCCO1']
        smiles_list23d = ['CC(SCNCC(=O)[O+]1PC=C[O+]=C1O)C(C)[SH]=O', 'CPPPCCS[SH](O)OSC', 'C=CCCC=CCC(OCC=O)SC', 'CC(C)OCO[SH](S)SSCCOOC1CCOCCCO1', 'O=C(C=CF)OCCO', 'CCC=CCCF', 'CPP=[PH](C=CO)[O+](C)NCCSO[SH](C)C', 'COCCSCC=O', 'C=C1P=C(CCC(F)Br)[SH](CC)[SH]2S[SH]2CP1', 'C=CC(=O)C(CC(C=COC)=COC)OCCCSC', 'COC(O)C=CF', 'CCOC(C)CCC(CF)C(O)=CSC(O)SC', 'CC(O)OC1CO[O+](C)N=C([OH+]CCF)O1', 'C[SH]1CS[O+]=[SH]2=CSC[SH]2C12PC=CC=C2O', 'C=COC(F)C(NSSCOOC=CCF)OSC', 'C=CO[SH]=CSC(=S)CC(=C)CF', 'CCC(CF)COOC(O)CCPCP', 'C[SH](SO)C(S)=CCF', 'CC(OOC=O)C(=O)C(Br)OPCS', 'CCOCC1CC(OCF)O[O+](C)CN2CSCCC2CCC1C']
        smiles_list23 = smiles_list23a + smiles_list23b + smiles_list23c + smiles_list23d

        smiles_list2 = smiles_list21 + smiles_list22 + smiles_list23



        counter = 1

        ori_graph = smiles2graph('COC1=CPCN(N)NC12PCNCC2(C)CC1CC=C(C)C(OC)CC1')
        node_feat_all = torch.from_numpy(ori_graph['node_feat']).to(torch.int64)
        edge_index_all = torch.from_numpy(ori_graph['edge_index']).to(torch.int64)
        edge_attr_all = torch.from_numpy(ori_graph['edge_feat']).to(torch.int64)
        num_nodes_all = torch.zeros(1,1)
        num_nodes_all = num_nodes_all + int(ori_graph['num_nodes'])
        num_edges_all = torch.zeros(1,1)
        num_edges_all = num_edges_all + int(ori_graph['edge_index'].size/2)

        for smiles in smiles_list2:
            counter = counter  + 1

            ori_graph = smiles2graph(smiles)
            node_feat = torch.from_numpy(ori_graph['node_feat']).to(torch.int64)
            edge_index = torch.from_numpy(ori_graph['edge_index']).to(torch.int64)
            edge_attr = torch.from_numpy(ori_graph['edge_feat']).to(torch.int64)
            num_nodes = torch.zeros(1, 1)
            num_nodes = num_nodes + int(ori_graph['num_nodes'])
            num_edges = torch.zeros(1, 1)
            num_edges = num_edges + int(ori_graph['edge_index'].size / 2)

            node_feat_all = torch.cat((node_feat_all, node_feat), dim=0)
            edge_index_all = torch.cat((edge_index_all, edge_index), dim=1)
            edge_attr_all = torch.cat((edge_attr_all, edge_attr), dim=0)
            num_nodes_all = torch.cat((num_nodes_all, num_nodes))
            num_edges_all = torch.cat((num_edges_all, num_edges))

        # test = pd.DataFrame(data=node_feat_all)  # 数据有三列，列名分别为one,two,three
        # print(test)
        # csv_file_name = '/home/gaohang/Researches/ChemGraph/dataset/cuihuaji/raw/backup' + '/node-feat.csv'
        # test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        edge_index_all = edge_index_all.transpose(0, 1)

        os.system('rm /home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_'+MOL_NAME+'/raw/*')

        folder_path = '/home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_'+MOL_NAME+'/raw/'

        # graph = Data(x, edge_index, edge_attr, num_nodes=num_nodes)

        # edge_counter = 0
        # graph_counter = 0
        # for edge_num in num_edge_list:
        #
        #     if int(edge_num[0]) > i:
        #         edge_delete = edge_counter + i
        #         edge.pop(edge_delete)
        #         edge_feat.pop(edge_delete)
        #         edge_counter = edge_counter + int(edge_num[0]) - 1
        #         num_edge_list[graph_counter] = str(int(edge_num[0]) - 1)
        #
        #     graph_counter = graph_counter + 1
        #
        # print("creating csv.gzs")
        #
        # folder_path = '/home/gaohang/Researches/ChemGraph/dataset/cuihuaji/raw/drop_' + str(i) + '/'
        # try:
        #     os.mkdir(folder_path)
        # except:
        #     print("create dir error!")
        #
        # # 下面这行代码运行报错
        # # name = ['one', 'two', 'three']
        edge_index_all_list = edge_ori + edge_index_all.tolist()
        test = pd.DataFrame(data=edge_index_all_list)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/edge.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )
        #
        edge_attr_all = edge_feat_ori + edge_attr_all.tolist()
        test = pd.DataFrame(data=edge_attr_all)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/edge-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )
        #
        graph_label = graph_label_ori + graph_label_ori[:counter]
        test = pd.DataFrame(data=graph_label)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/graph-label.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        node_feat_all = node_feat_ori + node_feat_all.tolist()
        test = pd.DataFrame(data=node_feat_all)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/node-feat.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        num_edges_all = num_edge_list_ori + num_edges_all.int().tolist()
        test = pd.DataFrame(data=num_edges_all)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/num-edge-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False )

        num_nodes_all = num_node_list_ori + num_nodes_all.int().tolist()
        test = pd.DataFrame(data=num_nodes_all)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = folder_path + '/num-node-list.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False  )

        os.system('gzip /home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_' + MOL_NAME + '/raw/*')

        graph_num = len(num_edge_list_ori)

        for i in range(counter):
            test_set_ori.append([str(i+graph_num)])

        # num_nodes_all = test_set_ori + num_nodes_all.int().tolist()
        test = pd.DataFrame(data=test_set_ori)  # 数据有三列，列名分别为one,two,three
        print(test)
        csv_file_name = '/home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_' + MOL_NAME + '//split/scaffold/test.csv'
        test.to_csv(csv_file_name, encoding='gbk', header = False , index = False  )

        os.system('rm /home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_' + MOL_NAME + '//split/scaffold/test.csv.gz')
        os.system('gzip /home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_' + MOL_NAME + '//split/scaffold/test.csv')

        os.system('rm /home/gaohang/Researches/GCU/adgcl-main-ori/original_datasets/ogbg_' + MOL_NAME + '/processed/*')

        print("create edge.csv.gzs")


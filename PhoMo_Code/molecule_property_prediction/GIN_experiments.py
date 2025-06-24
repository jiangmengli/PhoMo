import data_generate.create_csv_gz as create_data
import os
# import numpy as np
import feature_test
import graph_search_get
import torch
import csv
import both_test
import numpy as np

# def load_features(path, format = 'json'):
#     if format == 'json':
#         with open(path, 'r') as load_f:
#             load_dict = json.load(load_f)
#         print("loading json file:" + path)
#         features = load_dict
#     elif format == 'npy':
#         features = np.load(f'{path}')
#     else:
#         raise NotImplementedError
#
#     return features


if __name__ == '__main__':
    print("======= process start! =======\n")

    cmd = "rm .//GIN/GIN-PCQimp/first_substructure/first_substructure/*"
    os.system(cmd)

    cmd = "rm .//GIN/GIN-PCQimp/matching/*"
    os.system(cmd)

    cmd = "rm ./PC_experiment_json.json"
    os.system(cmd)

    cmd = "cp ./data_safecase/first_substructure/* .//GIN/GIN-PCQimp/first_substructure/first_substructure"
    os.system(cmd)

    cmd = "cp ./data_safecase/cuihuaji/* .//GIN/GIN-PCQimp/matching"
    os.system(cmd)

    cmd = "cp ./data_safecase/cuihuaji/PC_experiment_json.json ./PC_experiment_json.json"
    os.system(cmd)

    print("======= step 1: get graph matching outputs =======\n")

    skip=True

    if not skip:

        cmd = "cd ./GIN/GIN-PCQimp && /home/iscas/miniconda3/envs/gcl/bin/python /home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-graph-matching.py"

        os.system(cmd)

        cmd = "cp ./GIN/GIN-PCQimp/graph_match_judge.npy ./graph_match_judge.npy"

        os.system(cmd)

    print("======= step 2: test GIN =======\n")

    skip=False

    if not skip:

        cmd = "cd ./GIN/GIN-PCQimp && " \
              "/home/iscas/miniconda3/envs/gcl/bin/python " \
              "./main-genrate-pre.py"

        os.system(cmd)

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin --num_layers 5 "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gin_vFalse_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gin binary class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin --num_layers 5 "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gin_vFalse_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gin multi class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin  --num_layers 5"

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gin_vFalse_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gin binary class both test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin  --num_layers 5"

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gin_vFalse_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gin multi class both test acc is ", accs.mean(), " std is ", accs.std())

    print("======= step 3: test GIN-Virtual =======\n")

    skip=False

    if not skip:

        # cmd = "cd ./GIN/GIN-PCQimp && " \
        #       "/home/iscas/miniconda3/envs/gcl/bin/python " \
        #       "./main-genrate-pre.py"
        #
        # os.system(cmd)

        # cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "
        #
        # try:
        #     os.system(cmd)
        # except:
        #     pass
        #
        # os.system('export MKL_SERVICE_FORCE_INTEL=1')
        #
        # cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
        #       "/home/iscas/miniconda3/envs/gcl/bin/python " \
        #       "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual " \
        #       "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "
        #
        # os.system(cmd)

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual  --num_layers 5" \
                  "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gin_vTrue_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gin virtual binary class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual  --num_layers 5" \
                  "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gin_vTrue_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gin virtual multi class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual  --num_layers 5" \
                  "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gin_vTrue_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gin virtual binary class both test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual  --num_layers 5" \
                  "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gin_vTrue_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gin virtual multi class both test acc is ", accs.mean(), " std is ", accs.std())

    print("======= step 4: test GCN =======\n")

    skip=False

    if not skip:

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn " \
                  "--checkpoint_saved_dir ./checkpoint-gcn.pt"

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gcn binary class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn " \
                  "--checkpoint_saved_dir ./checkpoint-gcn.pt"

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gcn multi class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn " \
                  "--checkpoint_saved_dir ./checkpoint-gcn.pt"

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gcn binary class both test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn " \
                  "--checkpoint_saved_dir ./checkpoint-gcn.pt"

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gcn_vFalse_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gcn multi class both test acc is ", accs.mean(), " std is ", accs.std())

    print("======= step 5: test GCN-Virtual =======\n")

    skip=False

    if not skip:

        # cmd = "cd ./GIN/GIN-PCQimp && " \
        #       "/home/iscas/miniconda3/envs/gcl/bin/python " \
        #       "./main-genrate-pre.py"
        #
        # os.system(cmd)

        # cmd = "rm ./GIN/GIN-PCQimp/gin_vTrue_output_np.npy "
        #
        # try:
        #     os.system(cmd)
        # except:
        #     pass
        #
        # os.system('export MKL_SERVICE_FORCE_INTEL=1')
        #
        # cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
        #       "/home/iscas/miniconda3/envs/gcl/bin/python " \
        #       "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gin-virtual " \
        #       "--checkpoint_saved_dir ./checkpoint-gin-virtual.pt "
        #
        # os.system(cmd)

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn-virtual " \
                  "--checkpoint_saved_dir ./checkpoint-gcn-virtual.pt "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gcn virtual binary class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn-virtual " \
                  "--checkpoint_saved_dir ./checkpoint-gcn-virtual.pt "

            os.system(cmd)

            accs.append( feature_test.train("./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gcn virtual multi class feature test acc is ", accs.mean() , " std is ", accs.std())

        accs = []
        for i in range(0):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn-virtual " \
                  "--checkpoint_saved_dir ./checkpoint-gcn-virtual.pt "

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy", class_num='binary') )

        accs = np.array(accs)

        print("=== result === \n gcn virtual binary class both test acc is ", accs.mean(), " std is ", accs.std())

        accs = []
        for i in range(10):

            cmd = "rm ./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy "

            try:
                os.system(cmd)
            except:
                pass

            os.system('export MKL_SERVICE_FORCE_INTEL=1')

            cmd = "export MKL_SERVICE_FORCE_INTEL=1 && cd ./GIN/GIN-PCQimp && " \
                  "/home/iscas/miniconda3/envs/gcl/bin/python " \
                  "/home/iscas/GRL/ChemGraph/ChemGraph/WholeProcess2/GIN/GIN-PCQimp/main-ori.py --gnn gcn-virtual " \
                  "--checkpoint_saved_dir ./checkpoint-gcn-virtual.pt "

            os.system(cmd)

            accs.append( both_test.train("./GIN/GIN-PCQimp/gcn_vTrue_output_np.npy", class_num='multi') )

        accs = np.array(accs)

        print("=== result === \n gcn virtual multi class both test acc is ", accs.mean(), " std is ", accs.std())
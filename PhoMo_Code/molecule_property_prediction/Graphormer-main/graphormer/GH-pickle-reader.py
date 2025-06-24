import pickle
import wrapper


print("*******读取pickle文件********")

with open('/home/gaohang/Researches/ChemGraph/dataset/pyg_zinc/raw/bond_dict.pickle','rb') as test:
    data = pickle.load(test)
print(data)

# data.to_pickle('student.pickle')
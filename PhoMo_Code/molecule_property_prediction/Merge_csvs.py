import os
import csv
import tarfile
import gzip

def compress_to_gz(input_path, output_file):
    with tarfile.open(output_file, 'w:gz') as tar:
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_path)
                tar.add(file_path, arcname=arcname)



def merge_csvs(path1 = './dataset/cuihuaji_uncompressed', path2 = './dataset/cuihuaji_newfinded_uncompressed'):

    # 用于存储所有CSV文件的字典
    csv_data = {}

    cmd = "rm ./dataset/cuihuaji_merged/*"
    os.system(cmd)

    # 处理第一个文件夹
    for filename in os.listdir(path1):
        if filename.endswith('.csv'):
            with open(os.path.join(path1, filename), 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                data = [row for row in csv_reader if any(row)]  # 删除空行
                if filename in csv_data:
                    csv_data[filename].extend(data)
                else:
                    csv_data[filename] = data

    # 处理第二个文件夹
    for filename in os.listdir(path2):
        if filename.endswith('.csv'):
            with open(os.path.join(path2, filename), 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                data = [row for row in csv_reader if any(row)]  # 删除空行
                if filename in csv_data:
                    csv_data[filename].extend(data)
                else:
                    csv_data[filename] = data

    # 将数据写入新的CSV文件
    output_path = './dataset/cuihuaji_merged/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename, data in csv_data.items():
        output_file = os.path.join(output_path, filename)
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(data)

    print("CSV文件已处理、拼接并保存到output_folder中。")

    cmd = "gzip ./dataset/cuihuaji_merged/*.csv"
    os.system(cmd)

    print(f"已压缩")


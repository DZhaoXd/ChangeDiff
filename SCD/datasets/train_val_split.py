# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:05:41 2024

@author: 15642
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# 定义数据集路径
data_dir = '/data2/yjy/data/CNAM-CD(V1)'
A_dir = os.path.join(data_dir, 'A')
B_dir = os.path.join(data_dir, 'B')
C_dir = os.path.join(data_dir, 'label')

# 定义输出路径
# output_dir = 'output'
output_dir='/data2/yjy/data/CNAM-CD(V1)'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# 创建输出目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有文件名（假设A, B, C文件夹中的文件名相同）
file_names = os.listdir(A_dir)

# 将数据集划分为训练集和验证集
train_files, val_files = train_test_split(file_names, test_size=0.2, random_state=42)

def copy_files(file_list, source_dir, dest_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

# 将文件复制到训练集和验证集目录中
for file_name in train_files:
    # name,ext = os.path.splitext(file_name)
    # if 'a' in name:
    #     name = name.replace('a','')
    # file_name = name+ext
    os.makedirs(os.path.join(train_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'label'), exist_ok=True)
    shutil.copy(os.path.join(A_dir, file_name), os.path.join(train_dir, 'A', file_name))
    shutil.copy(os.path.join(B_dir, file_name), os.path.join(train_dir, 'B', file_name))
    shutil.copy(os.path.join(C_dir, file_name), os.path.join(train_dir, 'label', file_name))

for file_name in val_files:
    # name, ext = os.path.splitext(file_name)
    # if 'a' in name:
    #     name = name.replace('a', '')
    # file_name = name + ext
    os.makedirs(os.path.join(val_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'label'), exist_ok=True)
    shutil.copy(os.path.join(A_dir, file_name), os.path.join(val_dir, 'A', file_name))
    shutil.copy(os.path.join(B_dir, file_name), os.path.join(val_dir, 'B', file_name))
    shutil.copy(os.path.join(C_dir, file_name), os.path.join(val_dir, 'label', file_name))

print("数据集划分完成。")

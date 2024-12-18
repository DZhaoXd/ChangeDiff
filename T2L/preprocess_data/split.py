# -*- coding: utf-8 -*-
"""
Created on Sat May 18 05:05:07 2024

@author: 15642
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:01:37 2024

@author: 15642
"""

import os
import shutil
from PIL import Image
import numpy as np
import json
import tqdm



def Color2Index(ColorLabel, colormap2label=None, num_classes=None):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred, ST_COLORMAP=None):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]



def read_json_file_standard(file_path):
    try:
        with open(file_path, 'r') as file:
            # 读取 JSON 数据
            data = json.load(file)
            print("JSON 数据加载成功！")
            return data
    except FileNotFoundError:
        print("文件未找到：", file_path)
        return None
    except json.JSONDecodeError as e:
        print("JSON 解析错误：", e)
        return None
    
    
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # 逐行读取 JSON 数据
            lines = file.readlines()
            # 解析每一行的 JSON 数据
            data = [json.loads(line.strip()) for line in lines]
            print("JSON 数据加载成功！")
            return data
    except FileNotFoundError:
        print("文件未找到：", file_path)
        return None
    except json.JSONDecodeError as e:
        print("JSON 解析错误：", e)
        return None


def remove_top_level_directory(file_path):
    # 通过分隔符切分文件地址字符串
    parts = file_path.split(os.path.sep)
    
    # 删除最上级目录
    relative_path = os.path.sep.join(parts[1:])
    
    return relative_path


color_palette = [[255, 255, 255],
                   [0, 0, 255], 
                 [128, 128, 128], 
                 [0, 128, 0], 
                 [0, 255, 0], 
                 [128, 0, 0],
                 [255, 0, 0]]


axis_name = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']


# 定义原始文件夹和目标文件夹的路径
original_folder = 'label_merge'
target_folder = 'train_split'

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历原始文件夹中的所有子文件夹和文件
output_json_data = []
layout_save_path=  'train'
seg_save_path =  'second_layout'
class_num=len(axis_name)
class_pro = []

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(color_palette):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
            
            
for root, dirs, files in os.walk(original_folder):
    for file in tqdm.tqdm(files):
        # 检查文件是否以指定后缀结尾并且是png图片
        
        # 构建源文件路径和目标文件夹路径
        source_file = os.path.join(root, file)
        
        target_subfolder_path = os.path.join(target_folder, seg_save_path)
        # 如果目标子文件夹不存在，则创建它
        if not os.path.exists(target_subfolder_path):
            os.makedirs(target_subfolder_path)

        out_json = {} 
        out_json['file_name'] = file
        ## 处理并保存 attn_list 以及 text
        # 读取png图片并进行二值化处理
        img_name = file.split('.')[0]
        # img = np.array(Image.open(source_file).resize((512, 256), Image.NEAREST))
        img_rgb = np.array(Image.open(source_file))
        img = Color2Index(img_rgb, colormap2label, class_num)
        
        unique_ids = np.unique(img)
        if len(unique_ids) == 1:
            continue
        
        layout_target_path = os.path.join(target_folder, layout_save_path, file)
        if not os.path.exists(os.path.dirname(layout_target_path)):
            os.makedirs(os.path.dirname(layout_target_path)) 
        Image.fromarray(img_rgb).save(layout_target_path)
        
        ## 复制并保存 file_name
        # layout_source_path = source_file.replace('labelTrainIds', 'color')
        # layout_target_path = os.path.join(target_folder, layout_save_path, file)
        # if not os.path.exists(os.path.dirname(layout_target_path)):
        #     os.makedirs(os.path.dirname(layout_target_path)) 
        # shutil.copyfile(layout_source_path, layout_target_path)
        

        ### 统计类别向量
        class_pro_id = []
        for id_ in range(class_num):
            if id_ in unique_ids:
                id_per = np.ceil(np.average(img == id_)* 100)
                class_pro_id.append(id_per)
            else:
                class_pro_id.append(0)                
        out_json['class_pro'] = class_pro_id 
        class_pro.append(class_pro_id)
        ### For layout 训练
        text_prompt = 'A remote sensing image of'
        attn_list = []
        for train_id in unique_ids:
            train_id = int(train_id)
            if train_id == 0: continue
            id_per = np.ceil(np.average(img == train_id)* 100)
            save_image_name = 'mask_' + img_name + "_" + axis_name[train_id] + '.png' 
                
            target_file = os.path.join(target_subfolder_path, img_name, save_image_name)
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))                
            bin_mask = np.zeros_like(img)
            bin_mask[img == train_id] = 255
            Image.fromarray(bin_mask.astype(np.uint8)).save(target_file)
            
            class_ratio = str(int(id_per))
            text_prompt += ' ' + class_ratio + '%'

            text_prompt += ' ' +  axis_name[train_id]
            
            attn_list.append([axis_name[train_id], remove_top_level_directory(target_file)])

                
        out_json['text'] = text_prompt
        out_json['attn_list'] = attn_list
        output_json_data.append(out_json)

output_json_path= os.path.join(target_folder, layout_save_path, 'metadata.jsonl')
with open(output_json_path, 'w') as f:
    for d in output_json_data:
        f.write(json.dumps(d) + '\n')
        
### json 重新读取
### 
# re_load_json = load_json_file(output_json_path)
# for data in re_load_json:
#     print(data['class_pro'])
        

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 创建一个 T-SNE 模型
# tsne = TSNE(n_components=2, random_state=0)

# # 对输入向量进行降维
# embedded_vector = tsne.fit_transform(np.array(class_pro))

# # 提取降维后的坐标
# x = embedded_vector[:, 0]
# y = embedded_vector[:, 1]

# # 绘制散点图
# plt.scatter(x, y)
# plt.title('T-SNE Visualization')
# plt.show()

# plt.savefig('class_static_tsne.png', bbox_inches='tight', dpi=300)




import os
import shutil

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





color_palette = [[255, 255, 255],
                   [0, 0, 255], 
                 [128, 128, 128], 
                 [0, 128, 0], 
                 [0, 255, 0], 
                 [128, 0, 0],
                 [255, 0, 0]]

axis_name = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']



def id_image_to_rgb_id(img_ori):
    out_img = Image.fromarray(img_ori).convert('P')
    palette = []
    for cid, xx in enumerate(color_palette):
        # if cid not in METAINFO['class2trainid']: continue
        for x in xx:
            palette.append(x)
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    out_img.putpalette(palette)
    return out_img


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(color_palette):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

# 定义文件夹路径
label1_dir = 'label1'
label2_dir = 'label2'
label_merge_dir = 'label_merge'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(label_merge_dir):
    os.makedirs(label_merge_dir)


# label_merge_dir_id = 'label_merge_id'

# # 如果目标文件夹不存在，则创建它
# if not os.path.exists(label_merge_dir_id):
#     os.makedirs(label_merge_dir_id)



# 处理label1文件夹中的图片
for filename in os.listdir(label1_dir):
    if filename.endswith('.png'):  # 仅处理png文件
        new_filename = f'label1_{filename}'
        src_path = os.path.join(label1_dir, filename)
        dest_path = os.path.join(label_merge_dir, new_filename)
        shutil.copy(src_path, dest_path)
        
        # img_rgb = np.array(Image.open(dest_path))
        # img = Color2Index(img_rgb, colormap2label, len(axis_name))
        # img = id_image_to_rgb_id(Image.fromarray(img))
        # img.save(os.path.join(label_merge_dir_id, filename))
        
        
# 处理label2文件夹中的图片
for filename in os.listdir(label2_dir):
    if filename.endswith('.png'):  # 仅处理png文件
        new_filename = f'label2_{filename}'
        src_path = os.path.join(label2_dir, filename)
        dest_path = os.path.join(label_merge_dir, new_filename)
        shutil.copy(src_path, dest_path)
        



print("图片重命名并复制完成！")

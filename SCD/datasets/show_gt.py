# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 05:43:39 2024

@author: 15642
"""
import os
import shutil
from PIL import Image
import numpy as np
import json
import tqdm

cityspallete = [

]


def id_image_to_rgb_id(img_ori):
    out_img = Image.fromarray(img_ori).convert('P')
    palette = []
    for cid, xx in enumerate(cityspallete):
        # if cid not in METAINFO['class2trainid']: continue
        for x in xx:
            palette.append(x)
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(255)
    out_img.putpalette(palette)
    return out_img, out_img.convert('RGB')


gt_path = './gt'
save_path = './gt_show'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(gt_path):
    image = Image.open(os.path.join(gt_path, file))
    _, image_rgb = id_image_to_rgb_id(np.array(image))
    image_rgb.save(os.path.join(save_path, file.replace('gtFine_labelTrainIds', 'leftImg8bit' )))


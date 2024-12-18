# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 06:09:36 2024

@author: 15642
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import os
import json
import tqdm
from PIL import Image, ImageFilter
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
import re
import PIL

from PIL import Image, ImageDraw, ImageFont

def add_caption(image, caption, font_path="arial.ttf", font_size=80, text_color=(0, 0, 0), spacing=20):
    """
    Add caption to the image.

    Args:
    - image (PIL.Image.Image): Image object to add caption to.
    - caption (str): Caption text.
    - font_path (str): Path to the font file.
    - font_size (int): Font size for the caption.
    - text_color (tuple): RGB color tuple for the text color.
    - spacing (int): Spacing between image and caption.

    Returns:
    - PIL.Image.Image: Image object with caption added.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    text_width, text_height = draw.textsize(caption, font=font)
    width, height = image.size

    # Calculate position for the caption
    caption_position = ((width - text_width) / 2, height + spacing)

    # Create a new image with enough height to accommodate both image and caption
    new_image = Image.new("RGB", (width, height + text_height + spacing), color="white")

    # Paste the original image onto the new image
    new_image.paste(image, (0, 0))

    # Draw the caption on the new image
    draw = ImageDraw.Draw(new_image)
    draw.text(caption_position, caption, font=font, fill=text_color)

    return new_image


def create_image_collage(images, captions, font_path="arial.ttf", font_size=80, text_color=(0, 0, 0), spacing=20):
    """
    Create a collage by horizontally concatenating images with their captions.

    Args:
    - images (list): List of image paths.
    - captions (list): List of captions corresponding to the images.
    - font_path (str): Path to the font file.
    - font_size (int): Font size for the captions.
    - text_color (tuple): RGB color tuple for the text color.
    - spacing (int): Spacing between images.

    Returns:
    - PIL.Image.Image: Image object of the collage.
    """
    collage_parts = []

    # Iterate over each image and caption
    for image_path, caption in zip(images, captions):
        # Create an image with caption
        image_with_caption = add_caption(image_path, caption, font_path, font_size, text_color, 0)  # Use 0 spacing for individual images
        collage_parts.append(image_with_caption)

    # Calculate the total width and maximum height of the collage
    total_width = sum(part.width for part in collage_parts) + (len(collage_parts) - 1) * spacing
    max_height = max(part.height for part in collage_parts)

    # Create a blank canvas for the collage
    collage = Image.new("RGB", (total_width, max_height), color="white")

    # Paste each part of the collage onto the canvas with spacing
    x_offset = 0
    for part in collage_parts:
        collage.paste(part, (x_offset, 0))
        x_offset += part.width + spacing

    return collage

    
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
    
    
def load_txt_json_file(file_path):
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

def map_colors_to_ids(image):
    color_to_id =  [
                 [128, 128, 128], 
                 [0, 128, 0], 
                 [0, 255, 0], 
                 [0, 0, 255],
                 [255, 255, 255]]


        
    image = np.array(image)
    # 创建一个空的单通道ID图像
    id_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 将颜色映射表转换为NumPy数组
    if isinstance(color_to_id, dict):
        colors = np.array(list(color_to_id.values()))
    elif isinstance(color_to_id, list):
        colors = np.array(color_to_id)
    # 计算每个像素到所有颜色的L1距离
    distances = np.linalg.norm(image[:, :, np.newaxis, :] - colors, ord=1, axis=3)

    # 找到每个像素最近的颜色索引
    nearest_color_ids = np.argmin(distances, axis=2)

    # 将最近颜色的ID映射到图像中
    id_image = nearest_color_ids

    return id_image



def map_output_colors_to_ids(image):
    color_to_id =  [
                 [0, 155, 0], 
                 [255, 165, 0], 
                 [230, 30, 100], 
                 [0, 170, 240],
                 [255, 255, 255]]

    image = np.array(image)
    # 创建一个空的单通道ID图像
    id_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 将颜色映射表转换为NumPy数组
    if isinstance(color_to_id, dict):
        colors = np.array(list(color_to_id.values()))
    elif isinstance(color_to_id, list):
        colors = np.array(color_to_id)
    # 计算每个像素到所有颜色的L1距离
    distances = np.linalg.norm(image[:, :, np.newaxis, :] - colors, ord=1, axis=3)

    # 找到每个像素最近的颜色索引
    nearest_color_ids = np.argmin(distances, axis=2)

    # 将最近颜色的ID映射到图像中
    id_image = nearest_color_ids

    return id_image
    
    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_color_pallete(npimg, dataset='city'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            0, 155, 0,
            255, 165, 0,
            230, 30, 100,
            0, 170, 240,
            255, 255, 255,
        ]
        zero_pad = 256 * 3 - len(cityspallete)
        for i in range(zero_pad):
            cityspallete.append(255)
        out_img.putpalette(cityspallete)
    return out_img

def parse_class_percentages(prompt):
    # 使用正则表达式匹配出所有类别和比例
    pattern = r'(\d+)%\s(.+?)(?=\s\d+%|$)'
    matches = re.findall(pattern, prompt)

    # 提取出类别名和对应的比例并存储在字典中
    appeared_classes = {match[1]: int(match[0]) for match in matches}

    return appeared_classes


def random_in_derecse(input_list, N):
    A = copy.deepcopy(input_list)
    # 筛选出非零元素及其对应的索引
    non_zero_indices = [(i, num) for i, num in enumerate(A) if num != 0]
    
    # 找到非零元素中最小的三个数对应的索引
    min_three_indices = sorted(non_zero_indices, key=lambda x: x[1])[:3]
    min_three_idx = [index for index, value in min_three_indices]
    # min_non_zero = [value for index, value in min_three_indices]
    # 找到非零元素中最大的三个数对应的索引
    max_three_indices = sorted(non_zero_indices, key=lambda x: x[1], reverse=True)[:3]
    max_three_idx = [index for index, value in max_three_indices]
    max_three = [value for index, value in max_three_indices]
    
    # 计算N的值
    # 对最大的三个数进行随机减法操作
    total_to_decrease = N
    probabilities= np.array(max_three) / np.sum(np.array(max_three))

    while total_to_decrease > 0:
        # 随机选择要减去的数
        decrease_id = np.random.choice(max_three_idx, p=probabilities)
        # 计算减去的数
        decrease_amount = min(A[decrease_id] - 1, total_to_decrease)
        # 更新A中对应元素的值
        num = random.randint(1, decrease_amount)
        A[decrease_id] -= num
        # 更新剩余的减去的总量
        total_to_decrease -= num
    
    # 对最小的非零的三个数进行随机加法操作
    total_to_increase = N
    while total_to_increase > 0:
        # 随机选择要增加的数
        increase_id = np.random.choice(min_three_idx)
        # 计算增加的数
        increase_amount = min(total_to_increase, A[increase_id])
        # 更新A中对应元素的值
        num = random.randint(1, increase_amount)
        A[increase_id] += random.randint(1, increase_amount)
        # 更新剩余的增加的总量
        total_to_increase -= increase_amount
    return A


def modify_percentages(random_string, N=8):

    def parse_class_percentages_(prompt, class_name):
        # 使用正则表达式匹配出所有类别和比例
        pattern = r'(\d+)%\s(.+?)(?=\s\d+%|$)'
        matches = re.findall(pattern, prompt)
    
        # 提取出类别名和对应的比例并存储在字典中
        appeared_classes = {match[1]: int(match[0]) for match in matches}
        out_list = []
        for cname in class_name:
            if cname in appeared_classes:
                out_list.append(appeared_classes[cname])
            else:
                out_list.append(0)
        return out_list
    
    class_name = [ 'Farmland', 'Desert', 'Building', 'Water']

    init_static = parse_class_percentages_(random_string, class_name)
    final_static = random_in_derecse(init_static, N)

    # 构建修改后的字符串
    adjusted_elements = [f"{percentage}% {element}" for element, percentage in zip(class_name, final_static) if percentage > 0]
    adjusted_string = "A remote sensing image of " + " ".join(adjusted_elements)
    
    add_result = []
    sub_result = []
    for name, init, final in zip(class_name, init_static, final_static):
        diff = int(final - init)
        if diff > 0:
            add_result.append(f"+{diff}% {name}")
        elif diff < 0:
            sub_result.append(f"-{abs(diff)}% {name}")
    result_str = ", ".join(add_result)
    result_str += ", "
    result_str += ", ".join(sub_result)
    state = {}
    state['change_caption'] = result_str
    return adjusted_string, state



def add_random_class(class_names, prompt):
    # 解析字符串，获取已经出现的类别和其占比
    state = {}
    appeared_classes = parse_class_percentages(prompt)

    # 如果appeared_classes为空，则直接返回原字符串
    if not appeared_classes:
        return prompt, state

    # 从未出现的类别中随机选择一个类别
    unappeared_classes = [class_name for class_name in class_names if class_name not in appeared_classes]
    if len(unappeared_classes) == 0:
        state = None
        return prompt, state
    random_class = random.choice(unappeared_classes)
    random_percentage = random.randint(1, 3)
    
    # 从appeared_classes中选择一个占比大于增加类别占比的类别，如果没有则选择一个占比最大的类别
    valid_classes = [class_name for class_name, percentage in appeared_classes.items() if percentage > 0]
    if valid_classes:
        appeared_class_to_subtract = random.choice(valid_classes)
    else:
        appeared_class_to_subtract = max(appeared_classes, key=appeared_classes.get)

    # 随机减去出现的类别的占比，使其和增加的类别的占比保持一致
    if appeared_classes[appeared_class_to_subtract] <= random_percentage:
        random_percentage = appeared_classes[appeared_class_to_subtract]
        appeared_classes.pop(appeared_class_to_subtract)
    else:
        appeared_classes[appeared_class_to_subtract] -= random_percentage
        
    # 组成新的字符串
    merge_list = []
    not_insert = True
    for class_name, percentage in appeared_classes.items():
        if class_name == 'ignore': continue 
        class_id = class_names.index(class_name)
        incert_class_id = class_names.index(random_class)
        if class_id > incert_class_id and not_insert == True:
            merge_list.append("{}% {}".format(random_percentage, random_class))
            not_insert = False
        merge_list.append("{}% {}".format(percentage, class_name)) 
    if not_insert == True:
        merge_list.append("{}% {}".format(random_percentage, random_class))
    if 'ignore' in appeared_classes:
        merge_list.append("{}% ignore".format(appeared_classes['ignore'])) 
        
    new_prompt = "A remote sensing image of " + " ".join(merge_list)
    state['add'] = random_class
    state['add_num_per'] = random_percentage
    state['sub'] = appeared_class_to_subtract
    state['change_caption'] = "+ {}% {}, - {}".format(random_percentage, random_class, appeared_class_to_subtract)
    return new_prompt, state


def extract_class_proportions(class_name, prompt):
    # 初始化结果列表，将所有类别的占比初始化为0
    class_proportions = [0] * len(class_name)
    
    # 将 prompt 字符串按空格分割成单词列表
    words = prompt.split()
    
    # 遍历单词列表，查找包含类别名称和百分比的词组
    i = 0
    while i < len(words):
        if words[i] in class_name:
            # 如果词组中包含类别名称，则提取类别名称和百分比
            class_index = class_name.index(words[i])
            if i+1 < len(words):
                proportion = float(words[i + 1].strip('%')) 
                # 更新结果列表中对应类别的占比
                class_proportions[class_index] = proportion
            else:
                class_proportions[class_index] = 0
        i += 1
    
    return class_proportions

def find_most_similar_vector(vectors, input_vector):
    vectors = np.array(vectors)
    input_vector = np.array(input_vector)

    if np.max(vectors) > 1:
        vectors = vectors / 100
    if np.max(input_vector) > 1:
        input_vector = input_vector/100
    
    # 计算输入向量与所有向量之间的余弦相似度
    cosine_similarities = np.dot(vectors, input_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(input_vector))
    
    # 找到最相似向量的索引
    most_similar_index = np.argmax(cosine_similarities)
    
    return most_similar_index



#### class_define
class_name = [ 'Farmland', 'Desert', 'Building', 'Water' ]

# ls ~/.cache/huggingface/hub/
# cp train/results/TokenCompose/checkpoint-24000/unet/*  ~/.cache/huggingface/hub/models--layout--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/unet/


# LayoutDream diffusion 配置
model_id = "LayoutRS/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, local_files_only=True)
pipe = pipe.to(device)
height=512
width=512
in_channels=4
batch_size=1


# 初始化变量
output_json_data = []
source_num = 2000
add_num = 4
save_path = "./sample_results/add_class_landsat/"
# 创建目标文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)

## source json
source_base_path = 'train/data/CD_landsat/train'
source_txt_json_path = 'train/data/CD_landsat/train/metadata.jsonl'
source_json = load_txt_json_file(source_txt_json_path)
source_static = {}
for data in source_json:
    source_static[data['file_name']] = data['text']
random_elements = random.sample(source_json, source_num)


## prompt json
prompt_json_path = 'train/data/CD_landsat/train/metadata.jsonl'
prompt_json = load_txt_json_file(prompt_json_path)
random_prompts = random.sample(prompt_json, source_num)


## Sample A B B_1 B_2 B_3 B_4 ... ， 
## 并覆盖输出id的ignore的类别。

for num, random_element in enumerate(random_elements):
    # 如何定义初始的layout distribution
    A_prompt = random_prompts[num]['text']
    source_name = random_element['file_name'].split('.')[0]
    if 'label1' in source_name:
        B_name = random_element['file_name'].replace('label1', 'label2')
    else:
        B_name = random_element['file_name'].replace('label2', 'label1')
    B_prompt = source_static[B_name]
    
    ori_layout_a_path = os.path.join(source_base_path, random_element['file_name'])
    ori_layout_b_path = os.path.join(source_base_path, B_name)
    
    ori_layout_a = np.array(Image.open(ori_layout_a_path).resize((512, 512), Image.NEAREST))
    ori_layout_a = map_colors_to_ids(ori_layout_a)

    ori_layout_b = np.array(Image.open(ori_layout_b_path).resize((512, 512), Image.NEAREST))
    ori_layout_b = map_colors_to_ids(ori_layout_b)    

    
    Gen_prompt_list = []
    Gen_change_list = []
    Gen_layout_list = []
    Gen_layout_ID_list = []    

    generator = torch.Generator("cuda").manual_seed(num+66) # 定义随机seed，保证可重复性
    latents = torch.randn(
         (batch_size, in_channels, height // 8, width // 8),
         generator=generator, device=device
     )
    
    Layout_A = 0
    change_layout_ID_list = []
    replace_id = random.sample([0, 1], 1)[0]
    ori_replace = False
    prompt_success = True
    if np.random.rand() > 0.75:
        ori_replace = True

    for add_id in range(add_num):
        try:
            if add_id == 0:
                prompt = A_prompt
                state = {}
                state['change_caption'] = ""
            elif add_id == 1:
                prompt = B_prompt
                state = {}
                state['change_caption'] = ""
            else:
                prompt, ratio_state = modify_percentages(B_prompt, N=5)
                prompt, class_state = add_random_class(class_name, prompt)
                if class_state is not None:
                    state = class_state
                else:
                    state = ratio_state
            Gen_prompt_list.append(prompt)
            Gen_change_list.append(state['change_caption'])
        except:
            prompt_success = False
            break
        print(prompt)
        print(state['change_caption'])
        
        ALDM_prompt = ""
        
        image = pipe(prompt, latents=latents).images[0]
        
        ## 保存颜色版本layout
        image_name = source_name + '_{:05d}'.format(add_id) + '.png'
        color_file = os.path.join(save_path, 'layout_Color', image_name)
        if not os.path.exists(os.path.dirname(color_file)):
            os.makedirs(os.path.dirname(color_file))    
        image.resize((512, 512), resample=PIL.Image.NEAREST)
        #image.save(color_file)
        Gen_layout_list.append(image)
        
        ## 保存ID版本layout
        image_id_numpy = map_colors_to_ids(image)
        ###### 随机填充空白区域 即大于5的数
        # ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
        image_id_numpy[image_id_numpy>=len(class_name)] = replace_id
        image_id_numpy = Image.fromarray(image_id_numpy.astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
        image_id_numpy = np.array(image_id_numpy)
        ## copy-paste to image set.

        if ori_replace:
            if add_id == 0:
                image_id_numpy[ori_layout_a<len(class_name)] = ori_layout_a[ori_layout_a<len(class_name)]
            else:
                image_id_numpy[ori_layout_b<len(class_name)] = ori_layout_b[ori_layout_b<len(class_name)]
        ###### 
        image_id_name = source_name + '_{:05d}'.format(add_id) + '.png'
        id_file = os.path.join(save_path, 'layout_TrainIds', image_id_name)
        if not os.path.exists(os.path.dirname(id_file)):
            os.makedirs(os.path.dirname(id_file))  
        image_id = get_color_pallete(image_id_numpy).resize((512, 512), resample=PIL.Image.NEAREST)
        image_id.save(id_file)
        Gen_layout_ID_list.append(image_id)

        ### 保存 change_layout_ID_list
        if add_id == 0:
            import copy
            Layout_A_numpy = copy.deepcopy(image_id_numpy)
            init_change_label = np.ones_like(Layout_A_numpy) * (len(class_name)+1)
            change_layout_ID_list.append(get_color_pallete(init_change_label))
        else:   
            change_mask = np.bitwise_xor(Layout_A_numpy, image_id_numpy)
            image_id_numpy[change_mask==0] = (len(class_name)+1) # ignore
            change_layout_ID_list.append(get_color_pallete(image_id_numpy))
        
        ## 保存ALDM_prompt
        output_json_data.append({'file_name': image_id_name, 'caption': ALDM_prompt, 'layout_name': image_name})
    
    if prompt_success:
        captions = ['refer image'] +  Gen_change_list[1:]
        print(captions)
        merge_save_path = os.path.join(save_path, 'layout_Color_merge', source_name + '.jpg')
        if not os.path.exists(os.path.dirname(merge_save_path)):
            os.makedirs(os.path.dirname(merge_save_path))  
        merged_image = create_image_collage(Gen_layout_list, captions, spacing=20, font_path="/data/zd/Fonts/TIMESI.TTF", font_size=45, text_color="black")
        merged_image.save(merge_save_path)
        
        merge_save_path = os.path.join(save_path, 'layout_TrainIds_merge', source_name + '.jpg')
        if not os.path.exists(os.path.dirname(merge_save_path)):
            os.makedirs(os.path.dirname(merge_save_path))         
        merged_image = create_image_collage(Gen_layout_ID_list, captions, spacing=20, font_path="/data/zd/Fonts/TIMESI.TTF", font_size=45, text_color="black")
        merged_image.save(merge_save_path)
    
        merge_save_path = os.path.join(save_path, 'change_id_merge', source_name + '_change.jpg')
        if not os.path.exists(os.path.dirname(merge_save_path)):
            os.makedirs(os.path.dirname(merge_save_path))         
        merged_image = create_image_collage(change_layout_ID_list, captions, spacing=20, font_path="/data/zd/Fonts/TIMESI.TTF", font_size=45, text_color="black")
        merged_image.save(merge_save_path)

    
## 保存ALDM_prompt
output_json_path= os.path.join(save_path, './output.json') 
with open(output_json_path, 'w') as file:
    json.dump(output_json_data, file)
# with open(output_json_path, 'w') as f:
#     for d in output_json_data:
#         f.write(json.dumps(d) + '\n')
     

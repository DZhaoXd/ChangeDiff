from PIL import Image
import numpy as np
import os


def check_pixel_values_in_folder(folder_path, valid_range=(0, 4)):
    valid_min, valid_max = valid_range
    all_images_valid = True

    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)

            # 打开图片并转换为numpy数组
            image = Image.open(image_path)
            image_np = np.array(image)

            # 检查是否所有像素值都在0到4的范围内
            if not np.all((image_np >= valid_min) & (image_np <= valid_max)):
                # print(f"Image {filename} has pixel values outside the range {valid_min}-{valid_max}.")
                all_images_valid = False
            else:
                pass
                # print(f"Image {filename} is valid.")

    if all_images_valid:
        print("All images have pixel values within the valid range.")
    else:
        print("Some images have pixel values outside the valid range.")


# 示例文件夹路径
# folder_path = 'path/to/your/folder'
folder_path = '/data2/yjy/data/Landsat-SCD/labelA'
# 调用函数
check_pixel_values_in_folder(folder_path)

# 示例文件夹路径



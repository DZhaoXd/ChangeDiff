import os
import json
from tqdm import tqdm

if __name__ == '__main__':
    root_path = '/data/yrz/repos/TokenCompose/data/cityscapes/leftImg8bit/train'
    caption = ' '.join(['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])
    ret = []
    for dirname, dirs, files in tqdm(os.walk(root_path)):
        # print(dirs)
        for file in files:
            data = {}
            full_path = os.path.join(dirname, file)
            assert os.path.isfile(full_path)
            data['img_path'] = full_path
            data['caption'] = [caption]
            if len(data)!=0 :ret.append(data)
    print('data size: \n'
          '{}'.format(len(ret)))
    output_path = './input_test' + '.json'
    with open(output_path, 'w') as f:
        json.dump(ret,f,indent=2)


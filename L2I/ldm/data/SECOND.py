import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
from torch.utils.data import DataLoader, Dataset
import torch


label_mapping = {
    0: [0, 0, 255],  # water
    1: [128, 128, 128],  # ground
    2: [0, 128, 0],  # low vegetation
    3: [0, 255, 0],  # tree
    4: [128, 0, 0],  # building
    5: [255, 0, 0],  # sports field
    6: [255, 255, 255]  # ignore
}

label_palette = [
    0, 0, 255,
    128, 128, 128,
    0, 128, 0,
    0, 255, 0,
    128, 0, 0,
    255, 0, 0,
    255, 255, 255
]

# ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
SECOND_dict = {
    0: 'water',
    1: 'ground',
    # 2: 'N.v.g. surface',
    2: 'low vegetation',
    3: 'tree',
    4: 'building',
    5: 'sports field',
    6: 'ignore'
}


# Some words may differ from the class names defined in COCO-Stuff to minimize ambiguity
class SECONDBase(Dataset):
    def __init__(self,
                 json_file,
                 mode='train',
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 need_label_map=False,
                 white_area='random'
                 ):
        self.data_info = json_file
        with open(self.data_info, "r") as f:
            data_info_and_set = json.load(f)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.white_area = white_area
        self.flip_p = flip_p
        if mode == 'train':
            self.dataset = data_info_and_set['train_set']
        elif mode == 'test':
            self.dataset = data_info_and_set['test_set']
        elif mode == 'all':
            self.dataset = data_info_and_set['all']
        else:
            ValueError("Invalid mode:{}".format(mode))
        print(data_info_and_set['dataset_info'])
        self.need_label_map = need_label_map
        self.data_info_and_set = data_info_and_set
        dataset_list = []
        for filename in self.dataset:
            img_and_label = (filename[0], filename[1])
            dataset_list.append(img_and_label)
            # label = (filename[0])
            # dataset_list.append(label)
            # if len(filename) >= 6:
            #     label = (filename[0], filename[1], filename[2], filename[3], filename[4], filename[5])
            #     dataset_list.append(label)
            # if len(filename) >= 4:
            #     label = (filename[0], filename[1], filename[2], filename[3])
            #     dataset_list.append(label)
        self.dataset = dataset_list

    def __len__(self):
        return len(self.dataset)

    def show_image(self, image):
        from matplotlib import pyplot as plt
        image_show = Image.fromarray(image.astype('uint8')).convert('P')
        plt.imshow(image_show)
        plt.show()

    def show_label(self, label, class_color_map=None):

        from matplotlib import pyplot as plt
        label_show = Image.fromarray(label.astype('uint8')).convert('P')
        if class_color_map is not None:
            label_show.putpalette(class_color_map)
        plt.imshow(label_show)
        plt.show()

    def label_encode_color(self, label):
        # encode the mask using color coding
        # return label: Tensor [3,h,w], (-1,1)
        from einops import rearrange
        self.color_map = np.array([
            [0, 0, 255],
            [128, 128, 128],
            [0, 128, 0],
            [0, 255, 0],
            [128, 0, 0],
            [255, 0, 0],
            [255, 255, 255],
            ], dtype=np.uint8)
        label_ = np.copy(label)
        # label_[label_ == 255] = 26  #change to class e.g.:26
        label_ = np.array(label_).astype(np.uint8)
        label_ = self.color_map[label_]
        label_ = rearrange(label_, 'h w c -> c h w')
        label_ = torch.from_numpy(label_)
        label_ = label_ / 255.0 * 2 - 1
        return label_

    def label_map_color2id(self, label, reverse=False):
        if not reverse:
            label_out = np.zeros(label.shape[:2])
            h, w = label.shape[:2]
            for v, k in label_mapping.items():
                # mask = (label.reshape(-1, 3) == k).reshape(h, w)
                mask = np.sum(label == k, axis=2) // 3
                # assert (np.unique(mask) == np.array([0, 3])).all()
                # print('mask:', np.unique(mask))
                # print((label == k).shape)
                label_out[mask == 1] = v
            # label_out = label[0]
        else:
            label_out = np.zeros((*label.shape[:2], 3))
            for k, v in label_mapping.items():
                label_out[label == k] = v
        return label_out

    def white_process(self, label, a):
        label_out = np.zeros(label.shape[:2])
        h, w = label.shape[:2]
        for i in range(h):
            for j in range(w):
                if label[i][j] == 6:
                    label_out[i][j] = a
                else:
                    label_out[i][j] = label[i][j]
        return label_out


    def get_ids_and_captions(self, labels):
        # class_ids = sorted(np.unique(labels.astype(np.uint8),axis=2))
        class_ids = np.unique(labels.astype(np.uint8))
        if class_ids[-1] == 6:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += SECOND_dict[class_ids[i]]  # ori code: text += Cityscapes_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        return text, class_ids_final

    def get_crop_patch(self, w, h, crop_size=512):
        rectangle_0 = crop_size
        rectangle_1 = crop_size

        start_0 = np.random.randint(0, w - rectangle_0) if w > rectangle_0 else 0
        start_1 = np.random.randint(0, h - rectangle_1) if h > rectangle_1 else 0

        return (start_0, start_0 + rectangle_0, start_1, start_1 + rectangle_1)

    def combine_function_1(self, label: np.ndarray, label_aux: np.ndarray, range=None):
        label_B = label.copy()

        if range is None:
            range = [0.01, 0.5]

        rectangle_0 = int(label.shape[0] * np.random.random() * (range[1] - range[0]) + range[0])
        rectangle_1 = int(label.shape[1] * np.random.random() * (range[1] - range[0]) + range[0])

        start_0 = np.random.randint(0, label.shape[0] - rectangle_0)
        start_1 = np.random.randint(0, label.shape[1] - rectangle_1)

        label_B[start_0:start_0 + rectangle_0, start_1:start_1 + rectangle_1] = label_aux[start_0:start_0 + rectangle_0, start_1:start_1 + rectangle_1]

        return label_B

    def combine_function_2(self, label: np.ndarray, label_aux: np.ndarray):
        label_B = label.copy()

        class_ids_aux = np.unique(label_aux)
        object_id = np.random.choice(class_ids_aux)
        mask = label_aux == object_id

        label_B[mask] = label_aux[mask]

        return label_B

    def generate_B(self, label_map):
        rand_idx = np.random.randint(len(self.dataset))
        path_aux = self.dataset[rand_idx][1]
        pil_image_aux = Image.open(path_aux)
        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image_aux = pil_image_aux.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image_aux = pil_image_aux.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        label_aux = np.array(pil_image_aux).astype(np.uint8)
        if self.need_label_map:
            label_aux = self.label_map_color2id(label_aux)
        label_B = self.combine_function_1(label_map, label_aux)
        return label_B

    def __getitem__(self, i):
        example = dict()
        path = self.dataset[i][0]
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        path2 = self.dataset[i][1]
        pil_image2 = Image.open(path2)

        flip = random.random() < self.flip_p

        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        label = np.array(pil_image2).astype(np.uint8)

        # self.show_label(image)
        # self.show_label(label, class_color_map=change_color_map)

        # if self.size is not None:
        #     crop_slice = self.get_crop_patch(*label.shape, self.size)
        #     image = image[crop_slice[0]:crop_slice[1], crop_slice[2]:crop_slice[3]]
        #     label = label[crop_slice[0]:crop_slice[1], crop_slice[2]:crop_slice[3]]
        if self.need_label_map:
            # self.show_label(label)
            label = self.label_map_color2id(label)
        # if 4 in np.unique(label):
        # self.show_label(image)
        # self.show_label(label, class_color_map=label_palette)
        label_coded = self.label_encode_color(label)  # Tensor [3,h,w], (0,1)
        example["hint"] = label_coded
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["label"] = label.astype(np.float32)
        # print(np.unique(example['label']))
        # example["caption"] = text
        # example["class_ids"] = class_ids_final
        example["caption"], example["class_ids"] = self.get_ids_and_captions(label)
        example['label_B'] = self.generate_B(label)
        example["caption_B"], example["class_ids_B"] = self.get_ids_and_captions(example['label_B'])
        example["img_name"] = path.split("/")[-1]

        return example


class SECONDTrain(SECONDBase):
    def __init__(self, **kwargs):
        super().__init__(mode='train', **kwargs)


class SECONDTest(SECONDBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, mode='test', **kwargs)


class SECONDAll(SECONDBase):
    def __init__(self, **kwargs):
        super().__init__(mode='all', **kwargs)


class SECONDABbase(SECONDBase):
    def __init__(self, mode='all', gpu_info=None, **kwargs):
        super().__init__(mode=mode, **kwargs)
        if mode == 'train':
            self.dataset = self.data_info_and_set['train_set']
        elif mode == 'test':
            self.dataset = self.data_info_and_set['test_set']
        elif mode == 'all':
            self.dataset = self.data_info_and_set['all']
        else:
            ValueError("Invalid mode:{}".format(mode))
        dataset_list = []
        for filename in self.dataset:
            img_and_label = (filename[0], filename[1], filename[2], filename[3])
            dataset_list.append(img_and_label)
        self.dataset = dataset_list

        if gpu_info is not None:
            start_id = int(len(self.dataset) * (gpu_info[0] / gpu_info[1]))
            end_id = int(len(self.dataset) * ((gpu_info[0] + 1) / gpu_info[1]))
            print('start id:{} end id:{}'.format(start_id, end_id))
            self.dataset = self.dataset[start_id:end_id]
            print('gpu info : {} dataset length : {}'.format(gpu_info, len(self.dataset)))

        self._length = len(self.dataset)

    def __getitem__(self, i):
        example = dict()
        """open image and label : AB"""
        path0 = self.dataset[i][0]
        pil_image_A = Image.open(path0)
        if not pil_image_A.mode == "RGB":
            pil_image_A = pil_image_A.convert("RGB")
        path1 = self.dataset[i][1]

        pil_image_B = Image.open(path1)
        if not pil_image_B.mode == "RGB":
            pil_image_B = pil_image_B.convert("RGB")

        path2 = self.dataset[i][2]
        label_A = Image.open(path2)
        path3 = self.dataset[i][3]
        label_B = Image.open(path3)

        """ flip or resize"""
        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image_A = pil_image_A.resize((self.size, self.size), resample=self.interpolation)
            pil_image_B = pil_image_B.resize((self.size, self.size), resample=self.interpolation)
            label_A = label_A.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_B = label_B.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        if flip:
            pil_image_A = pil_image_A.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image_B = pil_image_B.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_A = label_A.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_B = label_B.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        """convert to ndarray"""
        pil_image_A = np.array(pil_image_A).astype(np.uint8)
        label_A = np.array(label_A).astype(np.uint8)
        pil_image_B = np.array(pil_image_B).astype(np.uint8)
        label_B = np.array(label_B).astype(np.uint8)

        if self.need_label_map:
            label_A = self.label_map_color2id(label_A)
            label_B = self.label_map_color2id(label_B)

        """return example"""
        example["image"] = (pil_image_A / 127.5 - 1.0).astype(np.float32)
        example["label"] = label_A.astype(np.float32)
        example["caption"], example["class_ids"] = self.get_ids_and_captions(label_A)

        example["image_B"] = (pil_image_B / 127.5 - 1.0).astype(np.float32)
        example['label_B'] = label_B.astype(np.float32)
        example["caption_B"], example["class_ids_B"] = self.get_ids_and_captions(label_B)
        example["img_name"] = path0.split("/")[-1]

        return example


class SECONDAbase(SECONDBase):
    # only A
    def __init__(self, mode='all', gpu_info=None, **kwargs):
        super().__init__(mode=mode, **kwargs)
        if mode == 'train':
            self.dataset = self.data_info_and_set['train_set']
        elif mode == 'test':
            self.dataset = self.data_info_and_set['test_set']
        elif mode == 'all':
            self.dataset = self.data_info_and_set['all']
        else:
            ValueError("Invalid mode:{}".format(mode))
        dataset_list = []
        for filename in self.dataset:
            img_and_label = (filename[0])
            dataset_list.append(img_and_label)
        self.dataset = dataset_list

        if gpu_info is not None:
            start_id = int(len(self.dataset) * (gpu_info[0] / gpu_info[1]))
            end_id = int(len(self.dataset) * ((gpu_info[0] + 1) / gpu_info[1]))
            print('start id:{} end id:{}'.format(start_id, end_id))
            self.dataset = self.dataset[start_id:end_id]
            print('gpu info : {} dataset length : {}'.format(gpu_info, len(self.dataset)))

        self._length = len(self.dataset)

    def __getitem__(self, i):
        example = dict()
        """open image and label : AB"""
        path0 = self.dataset[i]
        label_A = Image.open(path0)

        """ flip or resize"""
        flip = random.random() < self.flip_p
        if self.size is not None:
            label_A = label_A.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        if flip:
            label_A = label_A.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        """convert to ndarray"""
        label_A = np.array(label_A).astype(np.uint8)

        if self.need_label_map:
            label_A = self.label_map_color2id(label_A)
        """return example"""
        example["label"] = label_A.astype(np.float32)
        example["caption"], example["class_ids"] = self.get_ids_and_captions(label_A)
        example["img_name"] = path0.split("/")[-1]

        return example

class SECONDMbase(SECONDBase):
    # multi 6
    def __init__(self, mode='all', gpu_info=None, **kwargs):
        super().__init__(mode=mode, **kwargs)
        if mode == 'train':
            self.dataset = self.data_info_and_set['train_set']
        elif mode == 'test':
            self.dataset = self.data_info_and_set['test_set']
        elif mode == 'all':
            self.dataset = self.data_info_and_set['all']
        else:
            ValueError("Invalid mode:{}".format(mode))
        dataset_list = []
        for filename in self.dataset:
            # if len(filename) >= 4:
            #     label = (filename[0], filename[1], filename[2], filename[3])
            #     dataset_list.append(label)
            if len(filename) >= 6:
                label = (filename[0], filename[1], filename[2], filename[3], filename[4], filename[5])
                dataset_list.append(label)
        self.dataset = dataset_list


        if gpu_info is not None:
            start_id = int(len(self.dataset) * (gpu_info[0] / gpu_info[1]))
            end_id = int(len(self.dataset) * ((gpu_info[0] + 1) / gpu_info[1]))
            print('start id:{} end id:{}'.format(start_id, end_id))
            self.dataset = self.dataset[start_id:end_id]
            print('gpu info : {} dataset length : {}'.format(gpu_info, len(self.dataset)))

        self._length = len(self.dataset)

    def __getitem__(self, i):
        example = dict()
        """open multi label"""
        path0 = self.dataset[i][0]
        path1 = self.dataset[i][1]
        path2 = self.dataset[i][2]
        path3 = self.dataset[i][3]
        path4 = self.dataset[i][4]
        path5 = self.dataset[i][5]

        label_0 = Image.open(path0)
        label_1 = Image.open(path1)
        label_2 = Image.open(path2)
        label_3 = Image.open(path3)
        label_4 = Image.open(path4)
        label_5 = Image.open(path5)

        """ flip or resize"""
        flip = random.random() < self.flip_p
        if self.size is not None:
            label_0 = label_0.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_1 = label_1.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_2 = label_2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_3 = label_3.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_4 = label_4.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            label_5 = label_5.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        if flip:
            label_0 = label_0.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_1 = label_1.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_2 = label_2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_3 = label_3.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_4 = label_4.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            label_5 = label_5.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        """convert to ndarray"""
        label_0 = np.array(label_0).astype(np.uint8)
        label_1 = np.array(label_1).astype(np.uint8)
        label_2 = np.array(label_2).astype(np.uint8)
        label_3 = np.array(label_3).astype(np.uint8)
        label_4 = np.array(label_4).astype(np.uint8)
        label_5 = np.array(label_5).astype(np.uint8)

        if self.need_label_map:
            label_0 = self.label_map_color2id(label_0)
            label_1 = self.label_map_color2id(label_1)
            label_2 = self.label_map_color2id(label_2)
            label_3 = self.label_map_color2id(label_3)
            label_4 = self.label_map_color2id(label_4)
            label_5 = self.label_map_color2id(label_5)

        # if self.white_area == 'random':
        #     a = random.randint(0, 5)
        #     # print('white area id change to:', a)
        #     # print(path0, path1, path2, path3, path4, path5)
        #     label_0 = self.white_process(label_0, a)
        #     label_1 = self.white_process(label_1, a)
        #     label_2 = self.white_process(label_2, a)
        #     label_3 = self.white_process(label_3, a)
        #     label_4 = self.white_process(label_4, a)
        #     label_5 = self.white_process(label_5, a)

        """return example"""
        example["label0"] = label_0.astype(np.float32)
        example["caption0"], example["class_ids0"] = self.get_ids_and_captions(label_0)
        example["label1"] = label_1.astype(np.float32)
        example["caption1"], example["class_ids1"] = self.get_ids_and_captions(label_1)
        example["label2"] = label_2.astype(np.float32)
        example["caption2"], example["class_ids2"] = self.get_ids_and_captions(label_2)
        example["label3"] = label_3.astype(np.float32)
        example["caption3"], example["class_ids3"] = self.get_ids_and_captions(label_3)
        example["label4"] = label_4.astype(np.float32)
        example["caption4"], example["class_ids4"] = self.get_ids_and_captions(label_4)
        example["label5"] = label_5.astype(np.float32)
        example["caption5"], example["class_ids5"] = self.get_ids_and_captions(label_5)
        example["img_name0"] = path0.split("/")[-1]
        example["img_name1"] = path1.split("/")[-1]
        example["img_name2"] = path2.split("/")[-1]
        example["img_name3"] = path3.split("/")[-1]
        example["img_name4"] = path4.split("/")[-1]
        example["img_name5"] = path5.split("/")[-1]

        return example


if __name__ == '__main__':
    json_file = '/home/L2I/data/SECOND_train_concat_psd12_from_SCD_0.8.json'
    dataset = SECONDAll(json_file=json_file, need_label_map=True, flip_p=0, white_area='random')
    val_dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i, HR_img in enumerate(val_dataloader):
        print(i, HR_img['image'].shape)
        print(i, HR_img['hint'].shape)
        print(i, HR_img['label'].shape)
        print(i, HR_img['caption'])





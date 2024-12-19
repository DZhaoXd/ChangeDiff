import os.path

import torch.utils.data
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
import json


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img
    trg_img_np = trg_img

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


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


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_name='SECOND', data_root='data/', transform=None, json_file=None, mode='all'):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        if file_name == 'SECOND':
            self.num_classes = 7
            self.ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0],
                                [255, 0, 0]]
            self.ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
            self.height = 512
            self.width = 512
        elif file_name == 'LandsatSCD':
            self.num_classes = 5
            self.ST_COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]
            self.ST_CLASSES = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
            self.height = 416
            self.width = 416
        self.colormap2label = np.zeros(256 ** 3)
        for i, cm in enumerate(self.ST_COLORMAP):
            self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        # self.file_list = open(data_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        # self.pre_images = [data_root + '/' + dataset + '/im1/' + x for x in self.file_list]
        # self.post_images = [data_root + '/' + dataset + '/im2/' + x for x in self.file_list]
        # self.pre_gts = [data_root + '/' + dataset + '/label1/' + x for x in self.file_list]
        # self.post_gts = [data_root + '/' + dataset + '/label2/' + x for x in self.file_list]
        if not json_file:
            json_file = dataset + '.json'
        self.json_file = json_file
        self.dataset_info = json.load(open(data_root + '/' + dataset + '/' + json_file, 'r'))
        print(self.dataset_info['dataset_info'])
        print('use dataset mode:\t{}'.format(mode))
        self.file_list = []
        self.pre_images = []
        self.post_images = []
        self.pre_gts = []
        self.post_gts = []
        self.pseudo_pre_images = []
        self.pseudo_post_images = []
        self.pseudo_pre_gts = []
        self.pseudo_post_gts = []
        self.mode = mode

        for data_sample in self.dataset_info[mode]:
            self.pre_images.append(data_sample[0])
            self.post_images.append(data_sample[1])
            self.pre_gts.append(data_sample[2])
            self.post_gts.append(data_sample[3])
            self.pseudo_pre_images.append(data_sample[4])
            self.pseudo_post_images.append(data_sample[5])
            self.pseudo_pre_gts.append(data_sample[6])
            self.pseudo_post_gts.append(data_sample[7])
            assert os.path.basename(data_sample[0]) == os.path.basename(data_sample[1])
            assert os.path.basename(data_sample[0]) == os.path.basename(data_sample[2])
            assert os.path.basename(data_sample[4]) == os.path.basename(data_sample[5])
            assert os.path.basename(data_sample[4]) == os.path.basename(data_sample[6])
            self.file_list.append(os.path.basename(data_sample[0]))
        self.transform = transform
        if transform:
            self.train_transforms_all = A.Compose([
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(45, p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.RandomSizedCrop(min_max_height=(self.height, self.width),
                                  width=self.height, height=self.width, w2h_ratio=0.8, p=0.3),
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
            self.train_transforms_pre_image = A.Compose(
                [A.OneOf([
                    A.GaussNoise(p=1),
                    A.HueSaturationValue(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.Emboss(p=1),
                    A.MotionBlur(p=1),
                ], p=0.8)])
            self.train_transforms_post_image = A.Compose(
                [A.OneOf([
                    A.GaussNoise(p=1),
                    A.HueSaturationValue(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.Emboss(p=1),
                    A.MotionBlur(p=1),
                ], p=0.8)])
        self.normalize_image = A.Compose([
            A.Normalize()
        ], additional_targets={'image1': 'image'})
        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image', 'mask1': 'mask'})

    def __len__(self):
        return len(self.pre_images)

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        img = np.array(img)

        return img

    def get_train_data(self, pre_image, post_image, pre_label, post_label, transform=False):
        if self.transform:
            sample = self.train_transforms_all(image=pre_image, image1=post_image, mask=pre_label, mask1=post_label)
            pre_image, post_image, pre_label, post_label = (
                sample['image'], sample['image1'], sample['mask'], sample['mask1'])
            sample = self.train_transforms_pre_image(image=pre_image)
            pre_image = sample['image']
            sample = self.train_transforms_post_image(image=post_image)
            post_image = sample['image']
            if random.choice([0, 1]):
                pre_image, post_image = post_image, pre_image
                pre_label, post_label = post_label, pre_label

        sample = self.normalize_image(image=pre_image, image1=post_image)
        pre_image, post_image = sample['image'], sample['image1']
        sample = self.to_tensor(image=pre_image, image1=post_image, mask=pre_label, mask1=post_label)
        pre_image_tensor, post_image_tensor, pre_label_tensor, post_label_tensor = sample['image'].contiguous(), \
            sample['image1'].contiguous(), sample['mask'].contiguous(), sample['mask1'].contiguous()
        return pre_image_tensor, post_image_tensor, pre_label_tensor.unsqueeze(dim=0), post_label_tensor.unsqueeze(dim=0)

    def __getitem__(self, idx):
        pre_image_path = self.pre_images[idx]
        post_image_path = self.post_images[idx]
        pre_label_path = self.pre_gts[idx]
        post_label_path = self.post_gts[idx]
        #
        pre_image = self.load(pre_image_path)
        post_image = self.load(post_image_path)
        pre_label = self.load(pre_label_path)
        post_label = self.load(post_label_path)
        pre_label = Color2Index(pre_label, self.colormap2label, self.num_classes)
        post_label = Color2Index(post_label, self.colormap2label, self.num_classes)
        #

        pseudo_pre_image_path = self.pseudo_pre_images[idx]
        pseudo_post_image_path = self.pseudo_post_images[idx]
        pseudo_pre_label_path = self.pseudo_pre_gts[idx]
        pseudo_post_label_path = self.pseudo_post_gts[idx]
        #
        pseudo_pre_image = self.load(pseudo_pre_image_path)
        pseudo_post_image = self.load(pseudo_post_image_path)
        pseudo_pre_label = self.load(pseudo_pre_label_path)
        pseudo_post_label = self.load(pseudo_post_label_path)
        pseudo_pre_label = Color2Index(pseudo_pre_label, self.colormap2label, self.num_classes)
        pseudo_post_label = Color2Index(pseudo_post_label, self.colormap2label, self.num_classes)

        pre_image_tensor, post_image_tensor, pre_label_tensor, post_label_tensor = self.get_train_data(pre_image, post_image, pre_label, post_label, self.transform)
        # pseudo_pre_image = FDA_source_to_target_np(pseudo_pre_image, pre_image, L=0.01).astype('uint8')
        # pseudo_post_image = FDA_source_to_target_np(pseudo_post_image, post_image, L=0.01).astype('uint8')
        pseudo_pre_image_tensor, pseudo_post_image_tensor, pseudo_pre_label_tensor, pseudo_post_label_tensor = self.get_train_data(pseudo_pre_image, pseudo_post_image, pseudo_pre_label, pseudo_post_label,self.transform)

        return pre_image_tensor, post_image_tensor, pre_label_tensor, post_label_tensor, \
            pseudo_pre_image_tensor, pseudo_post_image_tensor, pseudo_pre_label_tensor, pseudo_post_label_tensor, self.file_list[idx]

    def get_img_info(self, idx):
        name = self.file_list[idx]
        return {"name": name}


if __name__ == '__main__':
    file_name = 'SECOND'
    data_root = '/data/yrz/repos/SCD/data/SECOND'
    # json_file = 'train_concat_from_FreeStyle.json'
    json_file = 'PSDImage_label_1-10/train_pair_StyleTransfer_1-15.json'
    dataset = Dataset('train', file_name=file_name, data_root=data_root, transform=True, json_file=json_file)

    for idata in range(len(dataset)):
        data = dataset.__getitem__(idata)

import argparse, os

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import json
import random
# from ldm.data.CITY import CITYABbase
from ldm.data.CITYAB import CITYABbase
from ldm.data.WHU import WHUAll, WHUABbase
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.COCO import COCO_dict
from ldm.data.ADE20K import ADE20K_dict
from ldm.data.HRSCD import HRSCD_dict
from ldm.data.xView2 import xView2ABbase
from ldm.data.SECOND import SECONDABbase
from torch.utils.data import DataLoader, Dataset


class COCOVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += COCO_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


class ADE20KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'annotations/validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += ADE20K_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


class HRSCDVal(Dataset):
    def __init__(self,
                 data_root,
                 json_file,
                 mode='test',
                 size=512,
                 interpolation="bicubic",
                 gpu_info=None
                 ):
        self.data_root = data_root
        self.data_info = json_file
        with open(self.data_info, "r") as f:
            data_info_and_set = json.load(f)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        if mode == 'train':
            self.dataset = data_info_and_set['train_set']
        elif mode == 'test':
            self.dataset = data_info_and_set['test_set']
        else:
            ValueError("Invalid mode:{}".format(mode))

        dataset_list = []
        for filename in self.dataset:
            img_and_label = (filename[0], filename[1])
            dataset_list.append(img_and_label)
        self.dataset = dataset_list

        if gpu_info is not None:
            start_id = int(len(self.dataset) * (gpu_info[0] / gpu_info[1]))
            end_id = int(len(self.dataset) * ((gpu_info[0] + 1) / gpu_info[1]))
            print('start id:{} end id:{}'.format(start_id, end_id))
            self.dataset = self.dataset[start_id:end_id]

        self._length = len(self.dataset)
        print(data_info_and_set['dataset_info'])
        self.data_info_and_set = data_info_and_set

    def __len__(self):
        return self._length

    def get_ids_and_captions(self, labels):
        # class_ids = sorted(np.unique(label.astype(np.uint8),axis=2))
        class_ids = np.unique(labels.astype(np.uint8))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += HRSCD_dict[class_ids[i]]  # ori code: text += Cityscapes_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        return text, class_ids_final

    def combine_function_1(self, label: np.ndarray, label_aux: np.ndarray, range=None):
        label_B = label.copy()

        if range is None:
            range = [0.1, 0.5]

        rectangle_0 = int(label.shape[0] * (np.random.random() * (range[1] - range[0]) + range[0]))
        rectangle_1 = int(label.shape[1] * (np.random.random() * (range[1] - range[0]) + range[0]))

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
        pil_image_aux = pil_image_aux.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label_aux = np.array(pil_image_aux).astype(np.uint8)
        # label_B = self.combine_function_1(label_map, label_aux)
        label_B = self.combine_function_2(label_map, label_aux)
        return label_B

    def __getitem__(self, i):
        example = dict()
        path = self.dataset[i][0]
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        example["img_name"] = path.split("/")[-1]

        path2 = self.dataset[i][1]
        pil_image2 = Image.open(path2)

        pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(pil_image2).astype(np.uint8)
        example["label"] = label.astype(np.float32)

        # example["caption"] = text
        # example["class_ids"] = class_ids_final
        example["caption"], example["class_ids"] = self.get_ids_and_captions(label)
        example['label_B'] = self.generate_B(label)
        example["caption_B"], example["class_ids_B"] = self.get_ids_and_captions(example['label_B'])

        return example


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    # 实例化 LatentDiffusion model
    model = instantiate_from_config(config.model)
    # 为 model 加载权重sd
    # m是一个列表，包含在加载状态字典时模型中缺失的键（参数）。
    # u是一个列表，包含加载状态字典时模型中未预期到的额外键（参数）。理想情况下，两者都是空的。
    m, u = model.load_state_dict(sd, strict=False)
    # “verbose” 参数通常是一个布尔值或整数，用来控制程序在执行时是否输出详细信息，以及输出信息的程度。
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/layout2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        default=False,
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-finetune_COCO.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="path to txt file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K", 'HRSCD', 'HRSCD_train', 'HRSCD_test', 'xView2', 'SECOND', 'WHU', 'CITY'],
        default="COCO"
    )
    parser.add_argument(
        '--gpu_info',
        type=str,
        default=None,
        required=False,
        help='use slice id in dataset'
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)  # 设置随机种子seed
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # 初始化模型并传入config中的参数，等效于下列代码
    # from ldm.models.diffusion.ddpm import LatentDiffusion
    # model = LatentDiffusion(**config.model.get("params", dict()))
    # model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # 加载Stable Diffusion模型

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'label_A'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'label_B'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'change'), exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    if opt.dataset == "COCO":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = COCOVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "ADE20K":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = ADE20KVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "HRSCD":
        class_color_map = [
            255, 255, 255,
            172, 15, 24,
            249, 215, 0,
            0, 119, 9,
            0, 96, 125,
            0, 11, 196,
        ]
        if opt.gpu_info is not None:
            opt.gpu_info = list(map(int, opt.gpu_info.split(',')))
        val_dataset = HRSCDVal(data_root=opt.data_root, json_file=opt.txt_file, gpu_info=opt.gpu_info)
    elif opt.dataset == "HRSCD_train":
        class_color_map = [
            255, 255, 255,
            172, 15, 24,
            249, 215, 0,
            0, 119, 9,
            0, 96, 125,
            0, 11, 196,
        ]
        if opt.gpu_info is not None:
            opt.gpu_info = list(map(int, opt.gpu_info.split(',')))
        val_dataset = HRSCDVal(data_root=opt.data_root, json_file=opt.txt_file, mode='train', gpu_info=opt.gpu_info)
    elif opt.dataset == "HRSCD_test":
        class_color_map = [
            255, 255, 255,
            172, 15, 24,
            249, 215, 0,
            0, 119, 9,
            0, 96, 125,
            0, 11, 196,
        ]
        if opt.gpu_info is not None:
            opt.gpu_info = list(map(int, opt.gpu_info.split(',')))
        val_dataset = HRSCDVal(data_root=opt.data_root, json_file=opt.txt_file, mode='test', gpu_info=opt.gpu_info)
    elif opt.dataset == "xView2":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = xView2ABbase(json_file=opt.txt_file, mode='train', gpu_info=opt.gpu_info, flip_p=0, need_label_map=True)
    elif opt.dataset == "WHU":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = WHUABbase(json_file=opt.txt_file, mode='train', gpu_info=opt.gpu_info, flip_p=0)
    elif opt.dataset == "LEVIR":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = xView2ABbase(json_file=opt.txt_file, mode='train', gpu_info=opt.gpu_info, flip_p=0, need_label_map=True)
    elif opt.dataset == "CITY":
        class_color_map = [0, 0, 0, 255, 255, 255, ]
        val_dataset = CITYABbase(json_file=opt.txt_file, mode='train', gpu_info=opt.gpu_info, flip_p=0)
    elif opt.dataset == "SECOND":
        class_color_map = [
            0, 0, 255,
            128, 128, 128,
            0, 128, 0,
            0, 255, 0,
            128, 0, 0,
            255, 0, 0
        ]
        if opt.gpu_info is not None:
            opt.gpu_info = list(map(int, opt.gpu_info.split(',')))
        val_dataset = SECONDABbase(json_file=opt.txt_file, gpu_info=opt.gpu_info, flip_p=0, need_label_map=True)
    else:
        raise ValueError
    zero_pad = 256 * 3 - len(class_color_map)
    for i in range(zero_pad):
        class_color_map.append(255)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    # start_code = None
    # if opt.fixed_code:
    #     start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        # torch.randn函数是PyTorch中用于生成具有正态分布（均值为0，标准差为1）的随机数的函数。

    def add_gaussian_noise_local_adaptive(image, block_size=32):
        noisy_image = image.clone()
        _, _, height, width = image.shape

        start_h = random.randint(0, height - block_size)
        start_w = random.randint(0, width - block_size)
        # patch = image[start_h: start_h + block_size, start_w: start_w + block_size]
        patch = image[:, :, start_h: start_h + block_size, start_w: start_w + block_size]

        mean = torch.mean(patch)
        stddev = torch.std(patch)
        noise = torch.randn_like(patch) * stddev + mean

        a = random.random()
        noisy_patch = patch * a + noise * (1 - a)
        # noisy_patch = patch  + noise
        # noisy_image[start_h: start_h + block_size, start_w: start_w + block_size] = noisy_patch
        noisy_image[:, :, start_h: start_h + block_size, start_w: start_w + block_size] = noisy_patch

        return noisy_image


    # 设置推理的精度
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    # 关闭梯度
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for idx, data in enumerate(val_dataloader):

                    start_code = None
                    if opt.fixed_code:
                        start_code = torch.randn([data["label"].shape[0], opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

                    # noisy_image = add_gaussian_noise_local_adaptive(start_code)

                    """A image generator"""
                    label = data["label"].to(device)
                    class_ids = data["class_ids"].to(device)
                    text = data["caption"]
                    # conditional prompt
                    c = model.get_learned_conditioning(text)
                    # print('caption:{} {}'.format(text, label.dtype))
                    # print('label:{} {}'.format(label.shape, label.dtype))
                    # print('condition:{} {}'.format(c.shape, c.dtype))
                    # unconditional prompt
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(data["label"].shape[0] * [""])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     label=label,
                                                     class_ids=class_ids,
                                                     batch_size=data["label"].shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    # decode_first_stage(VAE的decoder)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    # 对生成的图像数据进行归一化处理，将图像数据中的像素值限制在[0, 1]的范围内

                    for i in range(len(x_samples_ddim)):  # 保存A图片
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, 'A', f"{img_name}"))

                    """A label generator"""
                    for i in range(label.shape[0]):
                        label_A_i = label[i].cpu().numpy()
                        img_name = data["img_name"][i]
                        label_A_i = Image.fromarray(label_A_i.astype('uint8')).convert('P')
                        label_A_i.putpalette(class_color_map)
                        label_A_i.save(os.path.join(outpath, 'label_A', f"{img_name}"))

                    """B image generator"""
                    label_B = data["label_B"].to(device)
                    class_ids = data["class_ids_B"].to(device)
                    text = data["caption_B"]
                    c = model.get_learned_conditioning(text)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(data["label"].shape[0] * [""])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                    # noisy_image_B = add_gaussian_noise_local_adaptive(start_code)

                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     label=label_B,
                                                     class_ids=class_ids,
                                                     batch_size=data["label"].shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(len(x_samples_ddim)):
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, 'B', f"{img_name}"))

                    """B label generator"""
                    for i in range(label_B.shape[0]):
                        label_B_i = label_B[i].cpu().numpy()
                        img_name = data["img_name"][i]
                        label_B_i = Image.fromarray(label_B_i.astype('uint8')).convert('P')
                        label_B_i.putpalette(class_color_map)
                        label_B_i.save(os.path.join(outpath, 'label_B', f"{img_name}"))

                    # label_change = data["label_change"].to(device)
                    """change label generator"""
                    for i in range(label.shape[0]):
                        # label_change_i = label_change[i].cpu().numpy()
                        # img_name = data["img_name"][i]
                        # label_change_i = Image.fromarray(label_change_i.astype('uint8')).convert('P')
                        # label_change_i.putpalette(class_color_map)
                        # label_change_i.save(os.path.join(outpath, 'change', f"{img_name}"))
                        label_A_i, label_B_i = label[i].cpu().numpy(), label_B[i].cpu().numpy()
                        label_change = np.array(label_A_i != label_B_i).astype('uint8')
                        img_name = data["img_name"][i]
                        label_change = Image.fromarray(label_change.astype('uint8')).convert('P')
                        label_change.putpalette([0, 0, 0, 255, 255, 255])
                        label_change.save(os.path.join(outpath, 'change', f"{img_name}"))
                    # raise None


if __name__ == "__main__":
    main()

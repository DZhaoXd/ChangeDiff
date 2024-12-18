'''
Following code is adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import logging
import math
import os
import random
import shutil
import random
import itertools
import wandb

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from packaging import version


logger = get_logger(__name__, log_level="INFO")

if __name__ == '__main__':
    data_dir = 'data/citys_voc_ignore'
    dataset = load_dataset(
        "imagefolder",
        data_dir=data_dir,
        cache_dir=None,
    )
    print(dataset["train"])

## Environment Setup

Please follow the below steps:

```bash
conda create -n ChangeDiff python=3.8.5
conda activate ChangeDiff
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Setup

If you want to use your own data, please refer to [preprocess_data](preprocess_data/readme.md) for details.

### 1. Setup the Second Data dataset

```bash
cd train/data
# download Second dataset
# 
```
Form the second label as follows
```
train/data/
    label1/
    label2/
```
Run the preprocessing code:
```
python preprocess_data/merge_label.py
python preprocess_data/split.py
```
After this,  The users should from the following train/data directory:
```
train/data/
    coco_gsam_img/
        train/
			metadata.jsonl
            000000000142.jpg
            000000000370.jpg
            ...
    second_layout/
        label1_00001/
            mask_label1_00001_ground.png
            mask_label1_00001_low vegetation.png
            ...
        label1_00011/
            mask_label1_00011_building.png
            mask_label1_00011_ground.png
			mask_label1_00011_tree.png
            ...
        ...
```
## Training 
To run T2L, use the following command:

```bash
cd train
bash run.sh
```

The results will be saved under `train/results` directory.

## Sample layout from text 
To sample continuous layouts using T2L, use the following command:

```bash
cd infer
bash run_rs.sh
```

The results will be saved under `train/results` directory.


## License

This repository is released under the [Apache 2.0](LICENSE) license. 

## Acknowledgement

This code is built on [diffusers](https://github.com/huggingface/diffusers), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [VISOR](https://github.com/microsoft/VISOR), and [CLIP](https://github.com/openai/CLIP). We thank all these nicely open sourced code.

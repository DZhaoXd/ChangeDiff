

conda activate ChangeDiff

####################  second -scd

cp train/results/TokenCompose_second/checkpoint-80000/unet/* ~/.cache/huggingface/hub/models--LayoutRS--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/unet/

CUDA_VISIBLE_DEVICES=3 nohup python infer/add_class_infer_second.py > logs/add_class_infer_second.file 2>&1 &
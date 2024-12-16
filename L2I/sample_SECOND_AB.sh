echo $(pwd)
mkdir logs/

CUDA_VISIBLE_DEVICES=5 python LIS_AB.py --batch_size 8 \
    --config configs/stable-diffusion/v1-finetune_SECOND.yaml \
    --ckpt /home/L2I/logs/2023-12-25T15-34-07_exp_SECOND_from_SCD/checkpoints/last.ckpt \
    --dataset SECOND \
    --outdir outputs/SECOND_LIS_AB_From_SCD \
    --txt_file /home/L2I/data/sample_4.json \
    --data_root /home/L2I/data/SECOND \
    --plms



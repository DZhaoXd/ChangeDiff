echo $(pwd)
mkdir logs/

CUDA_VISIBLE_DEVICES=5 python LIS_AB.py --batch_size 8 \
    --config configs/stable-diffusion/v1-finetune_SECOND.yaml \
    --ckpt /home/L2I/logs/2023-12-25T15-34-07_exp_SECOND_from_SCD/checkpoints/last.ckpt \
    --dataset SECOND \
    --outdir outputs/SECOND_LIS_AB_From_SCD \
    --txt_file /home/L2I/data/SECOND_train_psd12_From_SCD_0.8.json \
    --data_root /home/L2I/data/SECOND \
    --plms

#CUDA_VISIBLE_DEVICES=2 nohup python LIS_AB.py --batch_size 8 \
#    --config configs/stable-diffusion/v1-finetune_SECOND.yaml \
#    --ckpt /home/L2I/logs/2023-12-27T18-11-58_exp_SECOND_from_SCD/checkpoints/last.ckpt \
#    --dataset SECOND \
#    --outdir outputs/SECOND_LIS_AB_From_SCD-1-10 \
#    --txt_file /home/L2I/data/SECOND_train_psd12_From_SCD_0.8.json \
#    --data_root /home/L2I/data/SECOND \
#    --plms \
#    --gpu_info 0,2 > logs/sample_SECOND_LIS_AB_From_SCD_log0.file 2>&1 &
#
#CUDA_VISIBLE_DEVICES=4 nohup python LIS_AB.py --batch_size 8 \
#    --config configs/stable-diffusion/v1-finetune_SECOND.yaml \
#    --ckpt /home/L2I/logs/2023-12-27T18-11-58_exp_SECOND_from_SCD/checkpoints/last.ckpt \
#    --dataset SECOND \
#    --outdir outputs/SECOND_LIS_AB_From_SCD-1-10 \
#    --txt_file /home/L2I/data/SECOND_train_psd12_From_SCD_0.8.json \
#    --data_root /home/L2I/data/SECOND \
#    --plms \
#    --gpu_info 1,2 > logs/sample_SECOND_LIS_AB_From_SCD_log1.file 2>&1 &


mkdir logs
#CUDA_VISIBLE_DEVICES=7 nohup python main.py --base configs/stable-diffusion/v1-finetune_SECOND.yaml \
#    -t \
#    --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
#    -n exp_SECOND_from_SCD \
#    --gpus 0, \
#    --data_root /home/L2I/data/SECOND \
#    --json_file /home/L2I/data/SECOND_train_concat_psd12_from_SCD_0.8.json >> logs/SECOND_allpsd12_512_from_SCD.file 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python main.py --base configs/stable-diffusion/v1-finetune_SECOND.yaml \
    -t \
    --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
    -n exp_SECOND_from_SCD \
    --gpus 0, \
    --data_root /home/L2I/data/SECOND \
    --no-test \
    --json_file /home/L2I/data/SECOND_train_concat_psd12_from_SCD_0.8.json > logs/SECOND_allpsd12_512_from_SCD.file 2>&1 &


CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_SECOND.yaml \
    -t \
    --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
    -n exp_SECOND_from_SCD \
    --gpus 0, \
    --data_root /home/L2I/data/SECOND \
    --no-test \
    --json_file /data/yrz/repos/FreeStyleNet/data/SECOND_train_concat_psd12_from_SCD_0.8.json


#mkdir logs/
#json_dir='Ori_image_label'
#json_dir='PSDImage_label_1-10'
json_dir='ori'
#mkdir logs/${json_dir}
#json_name='train_concat_from_FreeStyle_14_rule3'
#CUDA_VISIBLE_DEVICES=6 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_steps 20000 \
#                                            --json_file ${json_name}.json > logs/${json_name}.file 2>&1 &
#json_name='PSDImage_label_1-10'
#json_name='ori_image_label'
#CUDA_VISIBLE_DEVICES=7 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_ori_0.5'
#CUDA_VISIBLE_DEVICES=3 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#

#json_name='train_ori_0.25'
#CUDA_VISIBLE_DEVICES=3 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_ori_0.125'
#CUDA_VISIBLE_DEVICES=4 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_ori_0.0625'
#CUDA_VISIBLE_DEVICES=4 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_pair_Semi_1-15'
#CUDA_VISIBLE_DEVICES=5 nohup python train_semi.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_concat_from_FreeStyle_13_same'
#CUDA_VISIBLE_DEVICES=5 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_zip0.5_from_FreeStyle_13_same'
#CUDA_VISIBLE_DEVICES=5 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_zip0.25_from_FreeStyle_13_same'
#CUDA_VISIBLE_DEVICES=6 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_zip0.125_from_FreeStyle_13_same'
#CUDA_VISIBLE_DEVICES=5 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_zip0.0625_from_FreeStyle_13_same'
#CUDA_VISIBLE_DEVICES=6 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_pair_StyleTransfer_1-15'
#CUDA_VISIBLE_DEVICES=4 nohup python train_style.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

#json_name='train_concat_from_FreeStyle_14_rule1'
#CUDA_VISIBLE_DEVICES=1 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_concat_from_FreeStyle_14_rule2'
#CUDA_VISIBLE_DEVICES=1 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_concat_from_FreeStyle_14_rule3'
#CUDA_VISIBLE_DEVICES=2 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
#json_name='train_concat_from_FreeStyle_14_rule4'
#CUDA_VISIBLE_DEVICES=2 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#
### 最初始版本
#json_name='train_concat_from_FreeStyle_14_rule5'
#CUDA_VISIBLE_DEVICES=2 nohup python train.py --model_name A2Net  \
#                                            --file_name SECOND \
#                                            --inWidth 512 --inHeight 512 \
#                                            --save_dir ./weights/ \
#                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
#                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &

json_name='train_Syn_Image.json'
CUDA_VISIBLE_DEVICES=2 nohup python train.py --model_name A2Net  \
                                            --file_name SECOND \
                                            --inWidth 512 --inHeight 512 \
                                            --save_dir ./weights/ \
                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
                                            --logFile train_Syn_Image1.txt\
                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python train.py --model_name A2Net  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000 --json_file train_psd.json

#CUDA_VISIBLE_DEVICES=4 python train.py --model_name A2Net18  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000
#
#CUDA_VISIBLE_DEVICES=4 python train.py --model_name SSCDL  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000
#
#CUDA_VISIBLE_DEVICES=4 python train.py --model_name TED  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000
#
#CUDA_VISIBLE_DEVICES=4 python train.py --model_name BiSRNet  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000
#
#CUDA_VISIBLE_DEVICES=4 python train.py --model_name SCanNet  --file_name SECOND --inWidth 512 --inHeight 512 --lr 5e-4 --batch_size 8 --max_steps 20000
#
#
#python train.py --model_name A2Net  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000
#
#python train.py --model_name A2Net18  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000
#
#python train.py --model_name SSCDL  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000
#
#python train.py --model_name TED  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000
#
#python train.py --model_name BiSRNet  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000
#
#python train.py --model_name SCanNet  --file_name LandsatSCD --inWidth 416 --inHeight 416 --lr 5e-4 --batch_size 8 --max_steps 40000

# SCD

## 1. Introduction
Here is a framework for semantic change detection based on Pytorch.

Supported semantic change detection models:
- [x] [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814)
- [x] [SCanNet](https://arxiv.org/abs/2212.05245)
- [x] [TED](https://arxiv.org/abs/2212.05245)
- [x] [BiSRNet](https://ieeexplore.ieee.org/document/9721305)
- [x] [SSCDL](https://ieeexplore.ieee.org/document/9721305)

Supported semantic change detection datasets:
- [x] [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824) and Generation change detection data
- [x] [Landsat-SCD](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)

## 2. Usage
+ Prepare the data:
    - Download datasets [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824), and generate change detection data through L2I
    
+ Obtain the training json list:
    
    ```
    # You need to generate the corresponding JSON list according to your requirements.
    cd ./SCD/data/Second
    # You can choose this method according to your needs.
    python gen_img_list.py 
    # You also can choose this method according to your needs.
    python gen_Syn_img_list.py
    ```
    
+ Train:Assuming you have obtained the json list named list.json, you can run the following command to get the corresponding quantitative results.
    
    ```
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --model_name A2Net  \
         --file_name SECOND \
         --inWidth 512 --inHeight 512 \
         --save_dir ./weights/ \
         --lr 5e-4 --batch_size 8 --max_epochs 50\
         --json_file list.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &		
    ```
    
    




import os
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


def split_train_test(root, dir_pair=None, train_size=1.0, random_seed=42, save_path=None, prefix=None, flag=None, save=False,SyntheticID=False):
    if train_size >= 1.:
        train_val_file = "/data2/yjy/data/Landsat-SCD/train_list.txt"
    elif train_size <= 0.:
        train_val_file = "/data2/yjy/data/Landsat-SCD/val_list.txt"
    if not flag:
        flag = '.'
    if save_path is None:
        if prefix is None:
            save_path = '{}_{}.json'.format(os.path.basename(root), train_size)
        else:
            save_path = '{}_{}_{}.json'.format(os.path.basename(root), prefix, train_size)
    if dir_pair is None:
        dir_path = os.path.join(root)
        file_list = []
        for root, _, file_names in os.walk(dir_path):
            for filename in file_names:
                if flag not in filename: continue
                file_list.append(os.path.join(root, filename))
    elif isinstance(dir_pair, (tuple, list)):
        file_pair_list = []
        for dir_name in dir_pair:
            dir_path = os.path.join(root, dir_name)
            file_list = []
            for dir_root, _, file_names in os.walk(dir_path):
                for filename in file_names:
                    if flag not in filename: continue
                    assert os.path.isfile(os.path.join(dir_root, filename))
                    # with open(train_val_file, 'r') as file:
                    #     if filename in file.read():
                    file_list.append(os.path.join(dir_root, filename))
            file_pair_list.append(sorted(file_list))
        file_list = list(zip(*file_pair_list))
    elif isinstance(dir_pair, str):
        dir_path = os.path.join(root, dir_pair)
        file_list = []
        for root, _, file_names in os.walk(dir_path):
            for filename in file_names:
                if flag not in filename: continue
                file_list.append(os.path.join(root, filename))
    else:
        raise ValueError

    # file_list = [file_name for file_name in tqdm(os.listdir(dir_path))]
    if train_size <= 0.:
        train_set, test_set = [], file_list
    elif train_size >= 1.:
        train_set, test_set = file_list, []
    else:
        train_set, test_set = train_test_split(file_list, train_size=train_size, random_state=random_seed)
    print("All datasets:\t{}\n"
          "train_set:   \t{}\n"
          "test_set:    \t{}\n".format(len(file_list), len(train_set), len(test_set)))
    dataset_info = {'name': os.path.basename(root),
                    'dir_pair': dir_pair,
                    'root_path': root,
                    'all_number': len(file_list),
                    'train_size': train_size,
                    'train_number': len(train_set),
                    'test_number': len(test_set),
                    'random_seed': random_seed}
    ret = {'dataset_info': dataset_info, 'train': train_set, 'test': test_set, 'all': file_list}
    # if SyntheticID:
    #     for i,filepath in enumerate(ret['train']):
    #         path,filename = filepath.rsplit('/',1)
    #         name,ext = filename.rsplit()
    print(ret['dataset_info'])
    if save:
        with open(save_path, 'w') as f:
            json.dump(ret, f, indent=2)
    return ret


def concat_ret(ret_list, save_path=None):
    dataset_info = {'name': [],
                    'dir_pair': [],
                    'root_path': [],
                    'all_number': 0,
                    'train_size': [],
                    'train_number': 0,
                    'test_number': 0,
                    'random_seed': []}
    train_set = []
    test_set = []
    file_list = []

    for ret in ret_list:
        dataset_info['name'].append(ret['dataset_info']['name'])
        dataset_info['dir_pair'].append(ret['dataset_info']['dir_pair'])
        dataset_info['root_path'].append(ret['dataset_info']['root_path'])
        dataset_info['train_size'].append(ret['dataset_info']['train_size'])
        dataset_info['random_seed'].append(ret['dataset_info']['random_seed'])
        dataset_info['all_number'] += ret['dataset_info']['all_number']
        dataset_info['train_number'] += ret['dataset_info']['train_number']
        dataset_info['test_number'] += ret['dataset_info']['test_number']

        train_set.extend(ret['train'])
        test_set.extend(ret['test'])
        file_list.extend(ret['all'])
    ret = {'dataset_info': dataset_info, 'train': train_set, 'test': test_set, 'all': file_list}
    print(ret['dataset_info'])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(ret, f, indent=2)
    return ret


if __name__ == '__main__':
    # """psd version: SECOND for SCD"""
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # # sub_dir = ['im1', 'im2','label1','label2']
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # # sub_dir = ['im1', 'im2', 'pre_add_mean_predict', 'post_add_mean_predict','label1','label2']
    # sub_dir = ['im1', 'im2','label1','label2','im1_from_FreeStyle_1-10', 'im2_from_FreeStyle_1-10', 'label1', 'label2',]
    # prefix = None
    # flag = None
    # save_path = os.path.join(root, 'PSDImage_label_1-10/train_pair_StyleTransfer_1-15.json')
    # ret1 = split_train_test(root, sub_dir, prefix=prefix, flag=flag, save_path=save_path,save=True)

    # root = '/data/yrz/repos/SCD/data/SECOND/val'
    # sub_dir = ['im1', 'im2','label1','label2']
    # # sub_dir = ['im1', 'im2', 'psd1', 'psd2']
    # prefix = None
    # flag = None
    # save_path = os.path.join(root, 'val.json')
    # ret1 = split_train_test(root, sub_dir, prefix=prefix, flag=flag, save_path=save_path)

    # """psd version: SECOND for SCD"""
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # # sub_dir = ['im1', 'im2','label1','label2']
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # # sub_dir = ['im1', 'im2', 'pre_add_mean_predict', 'post_add_mean_predict','label1','label2']
    # sub_dir = ['im1', 'im2', 'label1', 'label2',]
    # prefix = None
    # flag = None
    # train_size = 0.0625
    # save_path = os.path.join(root, 'train_ori_0.0625.json')
    # split_train_test(root, sub_dir,train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)

    # """concat train.json"""
    #
    # image1, image2, label1, label2 = 'im1', 'im2', 'label1', 'label2'
    # ps_image1, ps_image2, ps_label1, ps_label2 = 'im1_from_FreeStyle_1-10', 'im2_from_FreeStyle_1-10', 'PSDImage_label_1-10/pre_reserved_by_rule5', 'PSDImage_label_1-10/post_reserved_by_rule5'
    #
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # sub_dir = [image1, image2, label1, label2]
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # # sub_dir = ['im1', 'im2', 'pre_add_mean_predict', 'post_add_mean_predict']
    # prefix = None
    # flag = None
    # save_path = None
    # train_size = 1.0 / 16.0
    # ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path)
    #
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # # sub_dir = ['im1', 'im2', 'label1', 'label2']
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # sub_dir = [image1, image2, ps_label1, ps_label2]
    # prefix = None
    # flag = None
    # save_path = None
    # ret2 = split_train_test(root, sub_dir, prefix=prefix, flag=flag, save_path=save_path)
    #
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # # sub_dir = ['im1', 'im2', 'label1', 'label2']
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # sub_dir = [ps_image1, ps_image2, label1, label2]
    # prefix = None
    # flag = None
    # save_path = None
    # ret3 = split_train_test(root, sub_dir, prefix=prefix, flag=flag, save_path=save_path)
    #
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # # sub_dir = ['im1', 'im2', 'label1', 'label2']
    # # sub_dir = ['im1', 'im2','psd1','psd2']
    # sub_dir = [ps_image1, ps_image2, ps_label1, ps_label2]
    # prefix = None
    # flag = None
    # save_path = None
    # ret4 = split_train_test(root, sub_dir, prefix=prefix, flag=flag, save_path=save_path)
    #
    # ret_list = [ret1, ret2, ret3, ret4]
    # # ret_list = [ret1,ret4]
    # # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_1-10/train_concat_from_FreeStyle_14_rule5.json'
    # # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_1-10/train_concat_from_FreeStyle_13_same.json'
    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_1-10/train_zip0.0625_from_FreeStyle_13_same.json'
    # concat_ret([ret1, ret3], save_path=save_path)

    # """psd version: SECOND for SCD"""
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # sub_dir = ['im1', 'im2', 'pre_add_mean_predict', 'post_add_mean_predict' ]
    # prefix = None
    # flag = None
    # train_size = 0.5
    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_no_ignore_3-4/train_ori_Image_meanPSDLabel_{}.json'.format(train_size)
    # ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)
    #
    # root = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_no_ignore_3-4'
    # sub_dir = ['im1_From_FreestyleNet_no_ignore', 'im2_From_FreestyleNet_no_ignore', 'label1_From_FreestyleNet_no_ignore', 'label2_From_FreestyleNet_no_ignore' ]
    # prefix = None
    # flag = None
    # train_size = 0.5
    # # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_no_ignore_3-4/train_PSDImageLabel_3-4_{}.json'.format(train_size)
    # save_path = None
    # save = False
    # ret2 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=save)
    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/PSDImage_label_no_ignore_3-4/train_concat_3-20_{}.json'.format(train_size)
    # concat_ret([ret1, ret2], save_path=save_path)

    # """psd version: SECOND for SCD"""
    # root = '/data/yrz/repos/SCD/data/SECOND/train'
    # sub_dir = ['im1', 'im2', 'label1', 'label2' ]
    # prefix = None
    # flag = None
    # train_size = 1
    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/ori/train_ori_Image_{}.json'.format(train_size)
    # ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)

    # root = '/data/yrz/repos/SCD/data/SECOND/val'
    # sub_dir = ['im1', 'im2', 'label1', 'label2' ]
    # prefix = None
    # flag = None
    # train_size = 0
    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/ori/test_ori_Image_{}.json'.format(train_size)
    # ret2 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)

    # save_path = '/data/yrz/repos/SCD/data/SECOND/train/ori/all_ori_Image_{}.json'.format(train_size)
    # concat_ret([ret1, ret2], save_path=save_path)

    # root = '/data2/yjy/data/Landsat-SCD'
    # sub_dir = ['A', 'B', 'labelA', 'labelB' ]
    # prefix = None
    # flag = None
    # train_size = 1
    # save_path = '/data2/ywj/SCD/json/LandsatSCD/train_ori_Image.json'
    # ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)
    #
    # root = '/data2/yjy/data/Landsat-SCD'
    # sub_dir = ['A', 'B', 'labelA', 'labelB']
    # prefix = None
    # flag = None
    # train_size = 0
    # save_path = '/data2/ywj/SCD/json/LandsatSCD/val_ori_Image.json'
    # ret2 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path,
    #                         save=True)
    #
    # save_path = '/data2/ywj/SCD/json/LandsatSCD/all_ori_Image_{}.json'.format(train_size)
    # concat_ret([ret1, ret2], save_path=save_path)

    root = '/data2/yjy/data/CNAM-CD_V1/train'
    sub_dir = ['A', 'B', 'label', 'label' ]
    prefix = None
    flag = None
    train_size = 1
    save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/train_ori_Image.json'
    ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)

    root = '/data2/yjy/data/CNAM-CD_V1/val'
    sub_dir = ['A', 'B', 'label', 'label' ]
    prefix = None
    flag = None
    train_size = 0
    save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/val_ori_Image.json'
    ret2 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path,
                            save=True)

    # save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/all_ori_Image.json'
    # concat_ret([ret1, ret2], save_path=save_path)
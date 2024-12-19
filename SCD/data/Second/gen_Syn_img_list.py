import os
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


def split_train_test(root, dir_pair=None, train_size=1.0, random_seed=42, save_path=None, prefix=None, flag=None, save=False,SyntheticID=False):
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
            file_list1 = []
            for dir_root, _, file_names in os.walk(dir_path):
                for filename in file_names:
                    if flag not in filename: continue
                    assert os.path.isfile(os.path.join(dir_root, filename))
                    # adding
                    if SyntheticID:
                        name,ext = filename.rsplit('_',1)
                        # if ext == '00000.png':
                        #     for i in range(3):
                        #         file_list1.append(os.path.join(dir_root, filename))
                        #     continue
                            ## 2
                        if ext == '00000.png':
                            continue
                        else:
                            file_list1.append(os.path.join(dir_root, name + "_00000.png"))
                            file_list.append(os.path.join(dir_root, filename))
                        # name,ext = filename.rsplit('_',1)
                        # if ext == '00000.png' or ext == '00001.png':
                        #     file_list1.append(os.path.join(dir_root,filename))
                        #     continue
                    else:
                        file_list.append(os.path.join(dir_root, filename))
            if SyntheticID:
                file_pair_list.append(sorted(file_list)+sorted(file_list1))
            else:
                file_pair_list.append(sorted(file_list))
            if SyntheticID:
                file_pair_list.append(sorted(file_list1)+sorted(file_list))
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

    # root = '/data2/yjy/FreeStyleNet/outputs/add_class_second_small_change_fixed_code_527'
    # root = '/data2/yjy/FreeStyleNet/outputs/add_class_second_small_change_fixed_code_602'
    # root = '/data2/yjy/FreeStyleNet/outputs/add_A_B_resample_604'
    root = '/data2/yjy/FreeStyleNet/outputs/add_class_CNAM_623'
    sub_dir = ['image', 'label' ]
    prefix = None
    flag = None
    train_size = 1
    save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/FreeStyleNet_Syn_Image.json'
    ret1 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True,SyntheticID=True)

    root = '/data2/yjy/data/CNAM-CD_V1/train'
    sub_dir = ['A', 'B', 'label', 'label' ]
    prefix = None
    flag = None
    train_size = 1
    save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/train_ori_Image.json'
    ret2 = split_train_test(root, sub_dir, train_size=train_size, prefix=prefix, flag=flag, save_path=save_path, save=True)

    save_path = '/data2/ywj/SCD/json/CNAM-CD_V1/all_Syn_Image_new_FreeStyleNet.json'
    concat_ret([ret1, ret2], save_path=save_path)

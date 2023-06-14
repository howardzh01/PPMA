import os
import sys

sys.path.append(os.path.abspath(''))
from pathlib import Path
import csv
import numpy as np
import random
import yaml
random.seed(42)

def get_prefix(file_name_suffix, dataset_name):
    if dataset_name == 'hmdb51':
        # 1. look at first 3 among phrase1_phrase2_phrase3...
        # 2. reduce length of prefix if any phrase is a string of length 1
        # push-up_or_shut_up_(IXK)_pushup_f_cm_np1_le_med_0
        prefix_arr = file_name_suffix.split('_')[:3]
        filtered_prefix_arr = prefix_arr[:1]
        for i in range(1, len(prefix_arr)):
            if len(prefix_arr[i]) == 1:
                break
            filtered_prefix_arr.append(prefix_arr[i])
        prefix = '_'.join(filtered_prefix_arr)
    elif dataset_name == 'ucf101':
        # v_BodyWeightSquats_g24_c03.avi
        prefix = '_'.join(file_name_suffix.split('_')[:-1])
    elif dataset_name == 'ikea':
        # pick_leg/GOPR0019_09318_09361_06.MP4
        prefix = '_'.join(file_name_suffix.split('_')[:1])
    elif dataset_name == 'minissv2':
        # 122448.webm
        prefix = file_name_suffix
    elif dataset_name == 'diving48':
        # DB4lpBDPnTY_00044.mp4
        prefix = '_'.join(file_name_suffix.split('_')[:1])
    elif dataset_name == 'uav':
        # P063S05G10B20H00UC072000LC021000A101R0_09101311.avi
        # file_name_suffix[:24] = P063S05G10B20H00UC072000
        prefix = file_name_suffix[:24]
    else:
        raise Exception(f"invalid dataset name: {dataset_name}")
    return prefix

def bucket_files(train_list, train_labels, dataset_name):
    # need to make sure train and val do not contain different clips of same video. each dataset has different rules
    # return label -> (prefix -> [filenames])
    bucket_dic = {}
    for file_name, label in zip(train_list, train_labels):
        prefix = get_prefix(file_name.split('/')[-1], dataset_name)
        if label not in bucket_dic:
            bucket_dic[label] = {}
        if prefix not in bucket_dic[label]:
            bucket_dic[label][prefix] = []
        bucket_dic[label][prefix].append(file_name)

    return bucket_dic

def divide_train_val(bucket_dic, train_prop=0.8):
    # bucket_dic = label -> (prefix -> [filenames]
    prefix_assignment = {}
    filename_assignment = {}
    for label in bucket_dic:
        for prefix in bucket_dic[label]:
            for file_name in bucket_dic[label][prefix]:
                if prefix in prefix_assignment:
                    filename_assignment[file_name] = (prefix_assignment[prefix], label)
                else:
                    prefix_assignment[prefix] = 'train_train' if random.random() < train_prop else 'train_val'
                    filename_assignment[file_name] = (prefix_assignment[prefix], label)
    return filename_assignment







if __name__ == '__main__':
    with open('config/experiments/dataset_catalog.yaml', 'r') as file:
        dataset_catalog = yaml.safe_load(file)
    for dataset in [
        'ikea', 'diving48','hmdb51',  'ucf101', 'minissv2', 'uav',
                    ]:

        vid_path = dataset_catalog[f'{dataset}_train_vids_path']
        vid_labels = dataset_catalog[f'{dataset}_train_labels_path']
        vid_prefix = dataset_catalog[f'{dataset}_prefix']
        video_path_arr, vid_label_arr = np.load(vid_path), np.load(vid_labels)
        bucket_dic = bucket_files(video_path_arr, vid_label_arr, dataset)
        filename_assignment = divide_train_val(bucket_dic, train_prop=0.8)
        print(dataset, sum([filename_assignment[f][0] == 'train_val' for f in filename_assignment])/len(filename_assignment))

        new_train_filename_path = [f for f in filename_assignment if filename_assignment[f][0] == 'train_train']
        new_train_label_path = [filename_assignment[f][1] for f in filename_assignment if filename_assignment[f][0] == 'train_train']
        new_val_filename_path = [f for f in filename_assignment if filename_assignment[f][0] == 'train_val']
        new_val_label_path = [filename_assignment[f][1] for f in filename_assignment if
                                filename_assignment[f][0] == 'train_val']
        print('train,val class count', len(set(new_train_label_path)), len(set(new_val_label_path)))
        if dataset == 'ucf101':
            print('saving')
            np.save(vid_path.replace('/splits/', '/val_splits/'), new_train_filename_path)
            np.save(vid_labels.replace('/splits/', '/val_splits/'), new_train_label_path)
            np.save(vid_path.replace('train', 'val').replace('/splits/', '/val_splits/'), new_val_filename_path)
            np.save(vid_labels.replace('train', 'val').replace('/splits/', '/val_splits/'), new_val_label_path)

        #
        # # if dataset == 'hmdb51':
        # print([f for f in filename_assignment if filename_assignment[f][0] == 'train_val'][:40])
        # print('\n\n\n\n')
        # print([f for f in filename_assignment if filename_assignment[f][0] == 'train_train'][:40])

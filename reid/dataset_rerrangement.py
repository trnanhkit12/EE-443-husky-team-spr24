import os
import numpy as np
from PIL import Image

# file naming
    # 0001_c1s1_001051_00.jpg
    # {pid}_{camid}s1_frameid_00.jpg
# gt format
    # 1,3,0,473,923,80,158
    # camid, pid, frameid, x, y, w, h
# extract bbox for bounding_box_train
    # use 2/3 pid for train
    # all camid
# extract bbox for test set
    # use 0th camid for query
    # use 1st camid for bounding_box_test

TRAIN_RATIO = 2/3           # the ratio being used to split pid for train and test

train_path = '/content/drive/MyDrive/EE443/final_proj/data/train'
exp_path = '/content/drive/MyDrive/EE443/final_proj/dataset_for_reid_train'
train_set1 = ['camera_0001', 'camera_0003']
train_set2 = ['camera_0011', 'camera_0013']
train_set3 = ['camera_0020', 'camera_0021']
dataset = [train_set1, train_set2, train_set3]

# data (np.array): the raw data corresponding to (train, query, or bbox_test)
# exp (str): the type of data being saved (train, query, or bbox_test)
# exp_pth (str): the exporting-to directory
#                (1 level higher than the (train, query, or bbox_test) directory)
def process_dir(data, exp, exp_pth):
    full_res_path = os.path.join(exp_pth, exp)
    if not os.path.exists(full_res_path):
        os.makedirs(full_res_path)
    for (camera_id, p_id, frame_id, x, y, w, h,) in data:
        camera_id_str = 'camera_' + camera_id.zfill(4)
        img_path = os.path.join(train_path, camera_id_str, frame_id.zfill(5) + '.jpg')
        img = Image.open(img_path)
        img_crop = img.crop((x-w/2, y-h/2, x+w/2, y+h/2))

        # 0001_c1s1_001051_00.jpg
        # {pid}_{camid}s1_frameid_00.jpg
        img_name = p_id.zfill(4) + f'_{camera_id}s1_' + frame_id.zfill(6) + '_00.jpg'
        img_saved_path = os.path.join(full_res_path, img_name)
        img_crop.save(img_saved_path, 'JPEG')


# given different datasets, produce different datasets that follows the format of Market_1501
def process_diff_dataset(datset):
    for idx, train_set in enumerate(datset):
        cur_exp_path = os.path.join(exp_path, f'train_set_{idx}')
        train_txt_path = [os.path.join(train_path, f'{folder}.txt') for folder in train_set]

        dat = np.vstack([np.genfromtxt(pth, dtype=str,delimiter=',') for pth in train_txt_path])

        pid = np.unique(dat[:, 2]).astype('str')
        tot_pid = len(pid)

        train_pid = np.arange(int(tot_pid * TRAIN_RATIO))
        train_idx = np.where(np.isin(dat[:, 1], train_pid))
        train_set = dat[train_idx]

        test_pid = np.arange(int(tot_pid * TRAIN_RATIO), tot_pid)
        camid = [int(cam_str[-4:])for cam_str in train_set1]

        query_idx = np.where((np.isin(dat[:, 1], test_pid)) and (np.isin(dat[:, 0], camid[0])))
        query_set = dat[query_idx]

        bbox_test_idx = np.where((np.isin(dat[:, 1], test_pid)) and (np.isin(dat[:, 0], camid[1])))
        bbox_test_set = dat[bbox_test_idx]

        print('Processing 0th dataset')
        print('     Processing bounding_box_train')
        process_dir(train_set, 'bounding_box_train', cur_exp_path)
        print('     Processing bounding_box_test')
        process_dir(bbox_test_set, 'bounding_box_test', cur_exp_path)
        print('     Processing query')
        process_dir(query_set, 'query', cur_exp_path)

process_diff_dataset(dataset)
# train_txt_path = [os.path.join(train_path, f'{folder}.txt') for folder in train_set1]
#
# dat = np.vstack([np.genfromtxt(pth, dtype=str,delimiter=',') for pth in train_txt_path])
#
# pid = np.unique(dat[:, 2]).astype('str')
# tot_pid = len(pid)
#
# train_pid = np.arange(int(tot_pid * TRAIN_RATIO))
# train_idx = np.where(np.isin(dat[:, 1], train_pid))
# train_set = dat[train_idx]
#
# test_pid = np.arange(int(tot_pid * TRAIN_RATIO), tot_pid)
# camid = [int(cam_str[-4:])for cam_str in train_set1]
#
# query_idx = np.where((np.isin(dat[:, 1], test_pid)) and (np.isin(dat[:, 0], camid[0])))
# query_set = dat[query_idx]
#
# bbox_test_idx = np.where((np.isin(dat[:, 1], test_pid)) and (np.isin(dat[:, 0], camid[1])))
# bbox_test_set = dat[bbox_test_idx]
#
# process_dir(train_set, 'bounding_box_train')
# process_dir(bbox_test_set, 'bounding_box_test')
# process_dir(query_set, 'query')

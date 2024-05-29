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

train_path = '/content/gdrive/MyDrive/EE443/final_proj/data/train'
exp_path = '/content/gdrive/MyDrive/EE443/final_proj/dataset_for_reid_train/market1501/Market-1501-v15.09.15'
train_set1 = ['camera_0001', 'camera_0003']
train_set2 = ['camera_0011', 'camera_0013']
train_set3 = ['camera_0020', 'camera_0021']
dataset = [train_set1, train_set2, train_set3]
dataset_test = train_set1

class dir_manager():
  def __init__(self, in_dir, out_dir):
    self.train_path = in_dir
    self.exp_path = out_dir
    self.cur_pid_range = 0
    self.test_in = 1
    self.test_out = 2
    
  # data (np.array): the raw data corresponding to (train, query, or bbox_test)
  # exp (str): the type of data being saved (train, query, or bbox_test)
  # exp_pth (str): the exporting-to directory
  #                (1 level higher than the (train, query, or bbox_test) directory)
  def process_dir(self, data, exp, exp_pth):
      full_res_path = os.path.join(exp_pth, exp)
      data = data.astype('int16')
      if not os.path.exists(full_res_path):
          os.makedirs(full_res_path)
      for (camera_id, p_id, frame_id, x, y, w, h,) in data:
          if self.test_in != self.test_out:
            print(f'in process_dir for loop {self.test_in}')
            self.test_in += 1
          camera_id_str = 'camera_' + camera_id.astype('str').zfill(4)
          img_path = os.path.join(self.train_path, camera_id_str, frame_id.astype\
                                  ('str').zfill(5) + '.jpg')
          img = Image.open(img_path)
          img_crop = img.crop((x, y, x+w, y+h))

          # 0001_c1s1_001051_00.jpg
          # {pid}_{camid}s1_frameid_00.jpg
          img_name = p_id.astype('str').zfill(4) + f'_c{camera_id}s1_'\
                     + frame_id.astype('str').zfill(6) + '_00.jpg'
          img_saved_path = os.path.join(full_res_path, img_name)
          img_crop.save(img_saved_path, 'JPEG')
      self.test_out += 1


  # given different datasets, produce different datasets that follows the format of Market_1501
  def process_diff_dataset(self, datset):
      for idx, dat_set in enumerate(datset):
          idx += 1 
          # cur_exp_path = os.path.join(self.exp_path, f'dat_set{idx}')
          cur_exp_path = os.path.join(self.exp_path)
          train_txt_path = [os.path.join(self.train_path, f'{folder}.txt') for folder in dat_set]

          dat = np.vstack([np.genfromtxt(pth, dtype=str,delimiter=',') for pth in train_txt_path])
          dat[:, 1] = (self.cur_pid_range + dat[:, 1].astype('int16')).astype('str')

          pid = np.unique(dat[:, 1]).astype('str')
          tot_pid = len(pid)
          self.cur_pid_range += np.max(pid.astype('int16'))
          
          tot_train_set_len = int(tot_pid * TRAIN_RATIO)
          train_pid = np.array([pid[i] for i in range(tot_train_set_len)]).astype('int16')
          train_idx = np.where(np.isin(dat[:, 1], train_pid.astype('str')))
          train_set = dat[train_idx]

          test_pid = np.array([pid[i] for i in range(tot_train_set_len, tot_pid)]).astype('int16')
          camid = np.array([int(cam_str[-4:]) for cam_str in dat_set])

          query_idx = np.where((np.isin(dat[:, 1], test_pid.astype('str'))) 
                                & (np.isin(dat[:, 0], camid[0].astype('str'))))
          query_set = dat[query_idx]

          bbox_test_idx = np.where((np.isin(dat[:, 1], test_pid.astype('str'))) 
                                    & (np.isin(dat[:, 0], camid[1].astype('str'))))
          bbox_test_set = dat[bbox_test_idx]
          print(f'Processing {idx} dataset')
          print('     Processing bounding_box_train')
          self.process_dir(train_set, 'bounding_box_train', cur_exp_path)
          print('     Processing bounding_box_test')
          self.process_dir(bbox_test_set, 'bounding_box_test', cur_exp_path)
          print('     Processing query')
          self.process_dir(query_set, 'query', cur_exp_path)

# process_diff_dataset(dataset)
manager = dir_manager(train_path, exp_path)
manager.process_diff_dataset(dataset)

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

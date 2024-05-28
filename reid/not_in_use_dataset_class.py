from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import sys
# import time
import numpy as np
import torchreid
from torchreid.reid.data import ImageDataset
# from torchreid.reid.utils import FeatureExtractor
# from torchreid.data import register_image_dataset

class NewDataset(ImageDataset):
    def __init__(self, camid, train_test_ratio, root='/content/drive/MyDrive/EE443/final_proj/data/train', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        # root is data root, dataset_dir is "train | val | test"
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.dataset_dir = self.root 
        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        # Generate train, query, and gallery lists

        train, query, gallery = self.train_process_dir(self.dataset_dir, camid, train_test_ratio)
        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    def train_process_dir(self, dataset_dir, camid, train_test_ratio):
        img_root = osp.join(dataset_dir, camid)
        txt_root = osp.join(dataset_dir, f'{camid}.txt')
        data = np.loadtxt(txt_root, delimiter=',').astype('int16')
        pid_range = np.unique(data[:,1])
        tot_pid = len(pid_range)
        num_train = int(tot_pid * train_test_ratio)
        train_pid = np.arange(num_train)
        test_pid = np.arange(num_train, tot_pid)
        train_idx = np.where(np.isin(data[:, 1], train_pid))
        test_idx = np.where(np.isin(data[:, 1], test_pid))
        train = self.get_img_camid_pid(img_root, data, train_idx)
        query = self.get_img_camid_pid(img_root, data, test_idx)
        gallery = self.get_img_camid_pid(img_root, data, test_idx, True)
        return train, query, gallery

    def get_img_camid_pid(self, img_root, data, idx, is_diff=False):
        idx = idx[0]
        dat = data[idx]
        pid = dat[:, 1]

        camid = data[0, 0]
        if is_diff:
            camid += 1
        camid = np.tile(camid, len(idx))

        frameid = np.unique(dat[:, 2]).astype('str')
        frameid = np.char.zfill(frameid, 5)
        frameid = np.char.add(frameid, '.jpg')
        img_pth = np.char.add(img_root, frameid)
        return np.hstack(img_pth, pid, camid).tolist()

# Factory function to create dataset classes
# def create_dataset_class(camid, train_test_ratio):
#     class_name = f"NewDataset_Cam{camid}"
#     return type(class_name, (NewDataset,), {"camid": camid, "train_test_ratio": train_test_ratio})

# # Create and register datasets for different camid values
# datasets = {}
# camid_lst = ['0001', '0003', '0011', '0013', '0020', '0021']
# for camid in camid_lst:
#     datasets[f"NewDataset_Cam{camid}"] = create_dataset_class(camid, train_test_ratio=0.7)

# for name, dataset_class in datasets.items():
#     torchreid.data.register_image_dataset(name, dataset_class)
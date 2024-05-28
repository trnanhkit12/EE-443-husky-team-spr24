import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser

import torchreid
from torchreid.reid.utils import FeatureExtractor


raw_data_root = '/content/gdrive/MyDrive/EE443/final_proj/data'

W, H = 1920, 1080
data_list = {
    # 'test': ['camera_0008', 'camera_0019', 'camera_0028']
    'test': ['camera_0028']
    # 'val': ['camera_0005', 'camera_0017', 'camera_0025']
}
sample_rate = 1 # because we want to test on all frames

det_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference/txt'
exp_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference'
reid_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'

val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

reid_extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=reid_model_ckpt,
    image_size=[256, 128],
    device='cuda' 
)   

for split in ['test']:
    for folder in data_list[split]:

        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        print(f"Extracting feature from {det_txt_path}")

        dets = np.genfromtxt(det_txt_path, dtype=str,delimiter=',')

        # start extracting frame features
        cur_frame = 0
        emb = np.array([None] * len(dets)) # initialize the feature array

        for idx, (camera_id, _, frame_id, x, y, w, h, score, _) in enumerate(dets):
            
            x, y, w, h = map(float, [x, y, w, h])
            frame_id = str(int(frame_id)) # remove leading space
            
            if idx % 1000 == 0:
                print(f'Processing frame {frame_id} | {idx}/{len(dets)}')
            
            img_path = os.path.join(raw_data_root, split, folder, frame_id.zfill(5) + '.jpg')
            img = Image.open(img_path)

            img_crop = img.crop((x-w/2, y-h/2, x+w/2, y+h/2))
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            feature = reid_extractor(img_crop).cpu().detach().numpy()[0]

            feature = feature / np.linalg.norm(feature)
            emb[idx] = feature

        # check embedding dimension
        emb_save_path = os.path.join(exp_path, f'{folder}.npy')
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        np.save(emb_save_path,emb)


# TODO 
# 1. You can try to extract features from the validation set since we have the ground truth labels
# 2. To determine how good the features, simply use the features to do the clustering and compare the clustering result with the ground truth labels

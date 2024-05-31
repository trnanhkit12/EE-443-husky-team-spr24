import os
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchreid.reid.utils import FeatureExtractor

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment


CamID = 'camera_0003'
raw_data_root = '/content/gdrive/MyDrive/EE443/final_proj/data'
# det_path = 'C:/Users/pt101/Desktop/EE-443-husky-team-spr24-master/runs/detect/inference/txt'
exp_path = '/content/gdrive/MyDrive/EE443/final_proj/cam_npy'
reid_first_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'
reid_second_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/log/osnet_ain/model/model.pth.tar-12'
gt_path = '/content/gdrive/MyDrive/EE443/final_proj/data/train'

det_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference/txt'
exp_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference'
reid_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'

data_list = {
    'train': ['camera_0003'],    
    # 'test' : ['camera_0008', 'camera_0019', 'camera_0028'],
    # 'val'  : ['camera_0005', 'camera_0017', 'camera_0025']
}


W, H = 1920, 1080
sample_rate = 1  # because we want to test on all frames

val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

reid_extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=reid_first_model_ckpt,
    image_size=[256, 128],
    device='cuda'
)

reid_extractor2 = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path=reid_second_model_ckpt,
    image_size=[256, 128],
    device='cuda'
)


for split in ['train']:
    for folder in data_list[split]:
        # det_txt_path = os.path.join(gt_path, f'{folder}.txt')                 # TESTING
        det_txt_path = os.path.join(det_path, f'{folder}_filt.txt')             # OFFICIAL FOR TESTING
        # det_txt_path = os.path.join(det_path, f'{folder}.txt')                # ORIGINAL
        print(f"Extracting feature from {det_txt_path}")

        dets = np.genfromtxt(det_txt_path, dtype=str, delimiter=',')
        cur_frame = 0
        emb = np.array([None] * len(dets))  # initialize the feature array
        emb2 = np.array([None] * len(dets))  # initialize the feature array 

        # for idx, (camera_id, _, frame_id, x, y, w, h) in enumerate(dets):     # TESTING
        for idx, (camera_id, _, frame_id, x, y, w, h, score, _) in enumerate(dets):   # ORIGINAL FOR DET RESULTS
            x, y, w, h = map(float, [x, y, w, h])
            frame_id = str(int(frame_id))  # remove leading space

            if idx % 1000 == 0:
                print(f'Processing frame {frame_id} | {idx}/{len(dets)}')

            img_path = os.path.join(raw_data_root, split, folder, frame_id.zfill(5) + '.jpg')
            img = Image.open(img_path)
            img_crop = img.crop((x, y, x + w, y + h))
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            feature = reid_extractor(img_crop).cpu().detach().numpy()[0]
            feature2 = reid_extractor2(img_crop).cpu().detach().numpy()[0]
            feature = feature / np.linalg.norm(feature)
            feature2 = feature2 / np.linalg.norm(feature2)
            emb[idx] = feature
            emb2[idx] = feature2
        
        reshaped_emb = np.stack(emb)
        print(reshaped_emb.shape)
        reshaped_emb2 = np.stack(emb2)
        print(reshaped_emb2.shape)
        shaped_emb = np.concatenate((reshaped_emb.T, reshaped_emb2.T)).T
        print(shaped_emb.shape)
        # emb_save_path = os.path.join(exp_path, f'{folder}_merged.npy')        # ORIGINAL
        emb_save_path = os.path.join(exp_path, f'{folder}_merged_filt.npy')
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        np.save(emb_save_path, shaped_emb)
        print(f"Saved merged feature to {emb_save_path}")


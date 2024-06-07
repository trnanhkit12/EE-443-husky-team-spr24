import os
import os.path as osp
import sys
import time

import numpy as np
from IoU_Tracker import tracker
from Processing import postprocess
from interpolation import kalman_interpolation
from interpolation import linear_interpolation

if len(sys.argv) > 1:
    conf_score = float(sys.argv[1])
else:
    conf_score = 0.5

if (len(sys.argv) > 2):
    interp = sys.argv[2]
else:
    interp = 'kalman'

raw_data_root = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/data'

W, H = 1920, 1080
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']
}
sample_rate = 1 # because we want to test on all frames
vis_flag = True # set to True to save the visualizations

exp_path = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/tracking/inference'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
det_path = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/detection/inference/txt'
emb_path = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/reid/inference'


for split in ['test']:
    for folder in data_list[split]:
        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        emb_npy_path = os.path.join(emb_path, f'{folder}.npy')
        tracking_txt_path = os.path.join(exp_path, f'{folder}.txt')

        #detection = np.loadtxt(det_txt_path, delimiter=',', dtype=None)
        detection = np.loadtxt(det_txt_path, delimiter=' ', dtype=None)
        embedding = np.load(emb_npy_path, allow_pickle=True)

        print(f"Getting bounding boxes from {det_txt_path} (number of detections: {len(detection)}")
        print(f"Getting features from {emb_npy_path} (number of embeddings: {len(embedding)}")


        camera_id = int(folder.split('_')[-1])
        print(f"Tracking on camera {camera_id}")

        # TODO
        # Can we remove some low score detection here? For example, removing all the detections with condifence score lower than 0.3?
        # Detection format: <camera ID>, <-1>, <Frame ID>, <x1>, <y1>, <w>, <h>, <confidence score>
        # Remember to change the embedding file too if you modify the detection!
        idx = detection[:, -2] >= conf_score
        detection = detection[idx, :]
        embedding = embedding[idx]

        mot = tracker()
        postprocessing = postprocess(number_of_people=50, cluster_method='agglomerative')

        # Run the IoU tracking
        tracklets = mot.run(detection, embedding)
        
        features = np.array([trk.final_features for trk in tracklets])

        # Run the Post Processing to merge the tracklets
        labels = postprocessing.run(features) # The label represents the final tracking ID, it starts from 0. We will make it start from 1 later.

        # interpolate missing detection of all tracks
        if (interp == 'kalman'):
            print("kalman interp")
            tracklets = kalman_interpolation(tracklets)
        elif (interp == 'linear'):
            print("linear interp")
            tracklets = linear_interpolation(tracklets)
    
        tracking_result = []

        print('Writing Result ... ')

        for i,trk in enumerate(tracklets):
            final_tracking_id = labels[i]+1 # make it starts with 1
            for idx in range(len(trk.boxes)):

                frame = trk.times[idx]
                x, y, w, h = trk.boxes[idx]
                
                result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera_id, final_tracking_id, frame, x-w/2, y-h/2, w, h )
                #result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera_id, final_tracking_id, frame, x, y, w, h )
                
                tracking_result.append(result)
        
        print('Save tracking results at {}'.format(tracking_txt_path))

        with open(tracking_txt_path,'w') as f:
            f.writelines(tracking_result)


# if __name__ == "__main__":

#     # camera = 74 # validation set
#     camera = 75 # test set

#     number_of_people = 5

#     result_path = 'baseline_result.txt'

#     # Load the data
#     detection = np.loadtxt('../detection.txt',delimiter=',',dtype=None)
#     embedding = np.load('../embedding.npy', allow_pickle=True)
#     inds = detection[:,0] == camera
#     test_detection = detection[inds]
#     test_embedding = embedding[inds]
#     sort_inds = test_detection[:, 1].argsort()
#     test_detection = test_detection[sort_inds]
#     test_embedding = test_embedding[sort_inds]

#     # TODO
#     # Can we remove some low score detection here? For example, removing all the detections with condifence score lower than 0.3?
#     # Detection format: <camera ID>, <-1>, <Frame ID>, <x1>, <y1>, <w>, <h>, <confidence score>
#     # Remember to change the embedding file too if you modify the detection!

#     mot = tracker()
#     postprocessing = postprocess(number_of_people,'kmeans')

    # # Run the IoU tracking
    # tracklets = mot.run(test_detection,test_embedding)

    # features = np.array([trk.final_features for trk in tracklets])

    # # Run the Post Processing to merge the tracklets
    # labels = postprocessing.run(features) # The label represents the final tracking ID, it starts from 0. We will make it start from 1 later.

    # tracking_result = []

    # print('Writing Result ... ')

    # for i,trk in enumerate(tracklets):
    #     final_tracking_id = labels[i]+1 # make it starts with 1
    #     for idx in range(len(trk.boxes)):

    #         frame = trk.times[idx]
    #         x1,y1,x2,y2 = trk.boxes[idx]
    #         x,y,w,h = x1,y1,x2-x1,y2-y1
            
    #         result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera, final_tracking_id, frame, x, y, w, h )
    
    #         tracking_result.append(result)
    
    # print('Save tracking results at {}'.format(result_path))

    # with open(result_path,'w') as f:
    #     f.writelines(tracking_result)

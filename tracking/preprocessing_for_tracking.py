import os
import numpy as np

# preprocessing_for_tracking.py
# This script filters for lines within the thresholds and export them in the
# same directory as the inputs

CONFIDENCE_THRESHOLD = 0.4
MIN_ASPECT_RATIO_THRESHOLD = 0.2
MAX_ASPECT_RATIO_THRESHOLD = 0.8

def valid_detection_idx(detection, confidence_threshold, min_ratio, max_ratio):
    """
    returns a boolean array to filter for lines within the thresholds from the detection results

    Args:
        detection (list or tuple): A detection represented as a list or tuple of floats
        confidence_threshold (float): The confidence threshold.
        min_ratio (float): Minimum acceptable aspect ratio.
        max_ratio (float): Maximum acceptable aspect ratio.

    Returns:
        combined_mask (np.array): a boolean array (as a mask) to retrieve lines within the thresholds
    """
    aspect_ratio = np.where(det[:, -3] != 0, det[:, -4] / det[:, -3], 0)
    gt_min_ratio = aspect_ratio > min_ratio
    gt_max_ratio = aspect_ratio < max_ratio
    gt_conf = detection[:,-2] > confidence_threshold
    combined_mask = gt_min_ratio & gt_max_ratio & gt_conf
    return combined_mask

# exp_path = "C:/Users/trnan/OneDrive/Desktop/uw/EE/EE 443/fin_proj"

result_txt = ['camera_0008', 'camera_0019', 'camera_0028']            # names of the detection result files (with .txt extension)
res_path = "/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference/txt"     # the detection result path

emb_npy = ['camera_0008', 'camera_0019', 'camera_0028']               # names of the npy files (without .npy extension)
emb_path = "/content/gdrive/MyDrive/EE443/final_proj/cam_npy"         # the corresponding feature extracted path

comb_lst = np.array([result_txt, emb_npy]).T

for result, emb in comb_lst:
    result_full_pth = os.path.join(res_path, f'{result}.txt')               # DETECTION RESULT TXT
    emb_npy_path = os.path.join(emb_path, f'{emb}.npy')                  # FEATURES EXTRACTED NPY
    det = np.genfromtxt(result_full_pth, dtype=float,delimiter=',')

    embedding = np.load(emb_npy_path, allow_pickle=True)

    idx = valid_detection_idx(det, CONFIDENCE_THRESHOLD,
                              MIN_ASPECT_RATIO_THRESHOLD,
                              MAX_ASPECT_RATIO_THRESHOLD)
    embedding = embedding[idx]
    det = det[idx].astype('str')

    emb_save_path = os.path.join(emb_path, f'{emb}_filt_conf_{CONFIDENCE_THRESHOLD}.npy')
    result_save_path = os.path.join(res_path, f'{result}_filt_conf_{CONFIDENCE_THRESHOLD}.txt')    
    if not os.path.exists(emb_npy_path):
        os.makedirs(emb_npy_path)
    np.savetxt(result_save_path, det,  fmt='%s')
    np.save(emb_save_path, embedding)
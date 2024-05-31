import os
import numpy as np
# preprocess.py
CONFIDENCE_THRESHOLD = 0.4
MIN_ASPECT_RATIO_THRESHOLD = 0.1
MAX_ASPECT_RATIO_THRESHOLD = 0.7
# preprocess.py


def is_valid_detection(detection, confidence_threshold, min_ratio, max_ratio):
  """
  cited from chatGPT
  Determines if a detection is valid based on confidence score and aspect ratio.

  Args:
    detection (list or tuple): A detection represented as a list or tuple of floats
    confidence_threshold (float): The confidence threshold.
    min_ratio (float): Minimum acceptable aspect ratio.
    max_ratio (float): Maximum acceptable aspect ratio.

  Returns:
    bool: True if the detection is valid, False otherwise.
  """
  _, _, _, _, _, width, height, confidence, _ = detection

  aspect_ratio = width / height if height != 0 else 0

  if ((confidence < confidence_threshold) or
    (aspect_ratio <= min_ratio) or
    (aspect_ratio >= max_ratio)):
    return False
  return True
filtered_gt_idx = np.array([])
res_path = "/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference/txt"
# result_txt = ['camera_0008', 'camera_0019', 'camera_0028']
result_txt = ['camera_0003']
for result in result_txt:
  result_full_pth = os.path.join(res_path, f'{result}.txt')
  result_filtered_pth = os.path.join(res_path, f'{result}_filt.txt')
  result_filtered_gt_pth = os.path.join(res_path, f'{result}_filt_gt.txt')
  with open(result_full_pth, 'r') as infile, open(result_filtered_pth, 'w') as outfile:
    for line in infile:
      detection = line.replace(',',' ')
      detection = list(map(float, detection.strip().split()))
      if is_valid_detection(detection, CONFIDENCE_THRESHOLD,
                            MIN_ASPECT_RATIO_THRESHOLD,
                            MAX_ASPECT_RATIO_THRESHOLD):
        filtered_gt_idx = np.append(filtered_gt_idx, np.array([True]))
        line_to_write = f'{int(detection[0])}, -1, {int(detection[2])}, {detection[3]},' \
                          f' {detection[4]}, {detection[5]}, {detection[6]}, {detection[7]},-1'
        outfile.write(line_to_write + '\n')
      else:
        np.append(filtered_gt_idx, np.array([False]))
  dat = np.genfromtxt(result_full_pth, dtype=str,delimiter=',').astype('str')
  data = dat[filtered_gt_idx.astype('int16')]
  print(data)
  np.savetxt(result_filtered_gt_pth, data,  fmt='%s')

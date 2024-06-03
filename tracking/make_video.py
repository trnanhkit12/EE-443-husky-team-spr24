import cv2
import os
import sys

if (len(sys.argv) > 1):
    camera = sys.argv[1]
else:
    camera = 'camera_0005'

if (len(sys.argv) > 2):
    video_name = sys.argv[2]
else:
    video_name = 'test.mp4'

image_folder = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/tracking/inference/vis/' + camera
video_name = '/content/drive/MyDrive/EE-443-husky-team-sam/EE-443-husky-team-spr24/tracking/inference/' + video_name

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 20.0, (width,height))

for i, image in enumerate(images):
    print(i)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

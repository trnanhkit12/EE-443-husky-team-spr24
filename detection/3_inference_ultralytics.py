import os
from ultralytics import YOLO


raw_data_root = '/content/gdrive/MyDrive/EE443/final_proj/data'

W, H = 1920, 1080
data_list = {
    # 'test': ['camera_0008', 'camera_0019', 'camera_0028'],
    'train': ['camera_0003']
}
sample_rate = 1 # because we want to test on all frames
vis_flag = True # set to True to save the visualizations

exp_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/inference'
model_path = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/runs/detect/train3/weights/best.pt'
det_model = YOLO(model_path)

# for split in ['test']:
for split in ['train']:
    for folder in data_list[split]:

        camera_img_folder = os.path.join(raw_data_root, split, folder)
        camera_img_list = os.listdir(camera_img_folder)
        camera_img_list.sort()

        camera_id = int(folder.split('_')[-1])
        lines_to_write = []

        if vis_flag:
            if not os.path.exists(os.path.join(exp_path, 'vis', folder)):
                os.makedirs(os.path.join(exp_path, 'vis', folder))

        for img_name in camera_img_list:
            frame_id = int(img_name.split('.')[0])

            results = det_model(camera_img_folder + '/' + img_name)
            boxes = results[0].boxes.xywh.cpu().numpy().tolist()
            confs = results[0].boxes.conf.cpu().numpy().tolist()

            if vis_flag:
                save_vis_img_path = os.path.join(exp_path, 'vis', folder, img_name)   
                results[0].save(filename=save_vis_img_path)
            
            # {camera_id}, {-1}, {frame_id}, {x}, {y}, {w}, {h}, {confidence}, {-1}
            for box, conf in zip(boxes, confs):
                x, y, w, h = box
                lines_to_write.append(f'{camera_id}, -1, {frame_id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, {conf:.3f},-1')

        # write the results to a file
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        with open(os.path.join(exp_path, 'txt', f'{folder}.txt'), 'w') as f:
            f.write('\n'.join(lines_to_write))
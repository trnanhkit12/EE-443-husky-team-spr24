# EE 443 2024 Challenge: Single Camera Multi-Object Tracking
## HUSKY TEAM

### TA: Chris Yang (cycyang), Chris Yin (c8yin)
### By: Samuel D. Profit, Kiet Tran, Anthony Chu, Zeqi Li

### Task Description
The EE 443 2024 Challenge: Single Camera Multi-Object Tracking aims to enhance the performance of object detection and tracking algorithms in single-camera environments. Participants will focus on improving detection models, ReID (Re-identification) models, and Multi-Object Tracking (MOT) algorithms to achieve superior object tracking accuracy.


### Detection

1. Install ultralytics (follow the [Quickstart - Ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics))

2. Download the `data.zip` from GDrive link provided in the Ed Discussion

Your folder structure should look like this:
```
├── data
│   ├── test
│   ├── train
│   └── val
├── detection
│   ├── 1_prepare_data_in_ultralytics_format.py
│   ├── 2_train_ultralytics.py
│   ├── 3_inference_ultralytics.py
│   └── ee443.yaml
```

4. Prepare the dataset into ultralytics format (remember to modified the path in the script)
```
python3 detection/1_prepare_data_in_ultralytics_format.py
```
After the script, your `ultralytics_data` folder should looke like this:
```
├── data
├── detection
├── ultralytics_data
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── val
│       ├── images
│       └── labels
```

4. Train the model using ultralytics formatted data (remember to modified the path in the script and config file `ee443.yaml`)
```
python3 detection/2_train_ultralytics.py
```
You model will be saved to `runs/detect/` with an unique naming.

5. Inference the model using the testing data (remember to modified the path in the script)
```
python3 detection/3_inference_ultralytics.py
```

### Re-ID

1. Install torchreid
2. Make a new directory ../market1501/Market-1501-v15.09.15 to prepare the training data for torchreid
3. Navigate to, modify the paths accordingly (see in-file comments/path variables), and run the file. After running, you should see the folder being populated with 3 new folders, in which there are cropped images named in the following format {pid}_{camid}s1_frameid_00.jpg
4. Configure the reid/train_model.py and run it to fine-tune the model
 
### Tracking

1. Configure the thresholds in the tracking/preprocessing_for_tracking.py and modify the path variables. Then, run the script. Filtered embedding features .npy and detection results .txt files will be generated in the same directory paths provided for the .npy and .txt files.
2. Configure the desired interpolation method, clustering method, the number of people, and the export path in the tracking/main.py. 
3. Run the script. The tracking result is now generated in the path you provided.

### Evaluation

1. Follow the readme in the evaluation directory.
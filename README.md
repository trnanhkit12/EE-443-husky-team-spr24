# EE 443 2024 Challenge: Single Camera Multi-Object Tracking

### TA: Chris Yang (cycyang), Chris Yin (c8yin)

### Task Description
The EE 443 2024 Challenge: Single Camera Multi-Object Tracking aims to enhance the performance of object detection and tracking algorithms in single-camera environments. Participants will focus on improving detection models, ReID (Re-identification) models, and Multi-Object Tracking (MOT) algorithms to achieve superior object tracking accuracy.

### Important Dates
- Release of the Challenge & Data: May 3rd, 2024
- Team Registration Due: May 8th | 11:59:59 pm
- Release of the Baseline Code: May 9th | 11:59:59 pm
- Final Submission (Results) Due: June 3rd | 11:59:59 pm
- Final Presentation (in-person): June 4th & June 6th
- Github & Final Report Due: June 7th | 11:59:59 pm

### Baseline Code for Detection

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


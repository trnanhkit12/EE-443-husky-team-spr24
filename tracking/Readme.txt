======================================================================================================================================================================
This is the improved tracking

To run the code, please install the required packages including numpy, sklearn, and scipy.

And then, directly run "python main.py". You should be able to obtain the tracking result.
======================================================================================================================================================================
Method Discription:

The improved method consists of three parts. 

1. A single camera tracking method using IoU for association. (IoU_Tracker.py)
2. A postprocessing clustering method that merge the tracklets together based on appearance. (Processing.py)
3. A postprocessing interpolation method that creates bounding boxes for missing frames. (Interpolation.py)
======================================================================================================================================================================
Before reading the code, please also get familiar with the idea of:

1. Hungarian algorithm (linear_sum_assignment) in IoU_Tracker.py, which is used for finding the best matching between two frames.
2. KMeans clustering in Processing.py, which is used to merge and cluster the tracklets with similar appearance together.
======================================================================================================================================================================
We also provide some hints and ideas for further modification to improve the baseline.

1. Can we filter out some low confidence score detection before tracking? It might be false positive detections.
1. Can we also use appearance for association in the single camera tracking stage? Perhaps make it into the Appearance_IoU_tracker (like DeepSORT)
2. Can we use different strategy for merging different tracklets in the post processing stage? Like Agglomerative Clustering in Scipy?
3. Can we remove some too short tracklets? They might be False Positive detections.
======================================================================================================================================================================

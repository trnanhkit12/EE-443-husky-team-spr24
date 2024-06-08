# Deep Sort 

This is the implemention of deep sort with pytorch.

The DeepSORT architecture consists of two main components: a Kalman filter and a deep appearance descriptor. The Kalman filter is used for predicting the position of objects in subsequent frames, handling the temporal dynamics of object movements. The deep appearance descriptor is a deep learning model trained to extract appearance features from detected objects, enabling the association of detections across frames based on visual similarity.

For detection association, we used the Hungarian algorithm to match detected objects in the current frame with tracked objects from previous frames. The matching cost was calculated based on a combination of the Mahalanobis distance between the predicted positions from the Kalman filter and the cosine distance between feature embeddings. For appearance feature extraction, we utilized a pretrained convolutional neural network (CNN) that was included in the provided implementation to compute a high-dimensional feature vector for each detected object. These feature vectors were then used to match detections across frames. By integrating deep appearance features, the tracker can maintain consistent identities for objects even when they reappear after occlusions or when viewed from different angles.

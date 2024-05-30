import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import numpy as np
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment



feature_path = 'C:/Users/pt101/Desktop/EE-443-husky-team-spr24-master/runs/reid/inference/camera_0001.npy'
groundtruth_path = 'C:/Users/pt101/Desktop/EE-443-husky-team-spr24-master/data/train/camera_0001.txt'

############################################ Cluster Number based on groundtruth  ############################################ 

# Path to the groundtruth
txt_file_path = groundtruth_path
# Read the file and load the content into a numpy array
data = np.loadtxt(txt_file_path, delimiter=',')
# Print the numpy array
print(data[0][2])
# Extract the second column (index 1)
second_column = data[:, 1]
# Find unique numbers in the second column
unique_numbers = np.unique(second_column)
# Count the number of unique numbers
unique_count = len(unique_numbers)
# Print the results
print("Unique numbers in the second column:", unique_numbers)
print("Number of cluster:", unique_count)


############################################ Cluster Label  ############################################ 
# Load the numpy files
emb_save_path = feature_path
emb = np.load(emb_save_path, allow_pickle=True)

reshaped_emb = np.stack(emb)
print(reshaped_emb.shape)  

print(f"Combined shape: {reshaped_emb.shape}")  # This should be (167188, 512)

# Number of clusters
num_clusters = unique_count  # You can choose the number of clusters based on your requirements

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reshaped_emb)

# Get cluster labels
cluster_labels = kmeans.labels_

############################################ Cluster Label from doing  ############################################ 

# Extract the PID (person ID) from ground truth
ground_truth_pids = second_column.astype(int)

# Compute the confusion matrix
confusion = confusion_matrix(ground_truth_pids, cluster_labels)

# Use the Hungarian algorithm to find the best matching
row_ind, col_ind = linear_sum_assignment(-confusion)

# Create a new mapping based on the best matching
new_cluster_labels = np.zeros_like(cluster_labels)
for i, j in zip(row_ind, col_ind):
    new_cluster_labels[cluster_labels == j] = i

# Evaluate the alignment quality
accuracy = np.mean(new_cluster_labels == ground_truth_pids)
print(f"Alignment accuracy: {accuracy:.4f}")

# Optionally, compute other metrics (precision, recall, F1 score)
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(ground_truth_pids, new_cluster_labels, average='weighted')
recall = recall_score(ground_truth_pids, new_cluster_labels, average='weighted')
f1 = f1_score(ground_truth_pids, new_cluster_labels, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

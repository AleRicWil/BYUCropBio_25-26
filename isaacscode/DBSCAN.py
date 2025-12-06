import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import ast



# Read the CSV file
df = pd.read_csv('filtered_flower_data.csv')

# Parse the GPS coordinates from string format to numpy arrays
def parse_gps(gps_string):
    return np.fromstring(gps_string.strip('[]'), sep=' ')

# Extract coordinates
coordinates = np.array([parse_gps(coord) for coord in df['GPSparsed']])

print(f"Loaded {len(coordinates)} data points")
print(f"Data shape: {coordinates.shape}")

# Optional: Standardize the data (recommended for DBSCAN)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(coordinates)

# Apply DBSCAN clustering
# eps: maximum distance between two samples to be considered neighbors
# min_samples: minimum number of samples in a neighborhood for a point to be a core point
# Start with eps=0.5 and min_samples=3, adjust based on your needs
dbscan = DBSCAN(eps=0.01, min_samples=3)
labels = dbscan.fit_predict(data_scaled)

# Get unique labels (clusters)
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nClustering Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Print cluster sizes
for label in sorted(unique_labels):
    if label != -1:
        count = list(labels).count(label)
        print(f"  Cluster {label}: {count} points")

# Visualize the results
plt.figure()

# Plot 1: Clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = 'black'
        marker = 'x'
        label_name = 'Noise'
        size = 6
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
        size = 10
    
    # Select points belonging to this cluster
    class_member_mask = (labels == label)
    xy = coordinates[class_member_mask]
    
    plt.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, 
                label=label_name, s=size, alpha=0.7, edgecolors='k', linewidths=0.5)

plt.title(f'DBSCAN Clustering Results\nClusters: {n_clusters}, Noise: {n_noise}')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
# plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.show()

# Add cluster labels to the dataframe
df['cluster'] = labels

# Save results to CSV
df.to_csv('clustered_data.csv', index=False)


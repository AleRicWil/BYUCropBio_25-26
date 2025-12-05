import numpy as np
import pandas as pd
import time
import ast
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from nn_v1_2 import associateData

# Earth approximation constants (for lat/lon to meters conversion)
METERS_PER_DEG_LAT = 111319.9  # Approximate meters per degree latitude

def convertGPS(GPSpoint):
    GPSpoint = ast.literal_eval(GPSpoint)  # Parse to list of strings
    GPSpoints = np.array([[float(x) for x in item.strip('[]').split()] for item in GPSpoint])  # To (n,2) array
    return GPSpoints

def updatelandmark(mu, Sigma, z, R):
    K = Sigma @ np.linalg.inv(Sigma + R)
    mu_new = mu + K @ (z - mu)
    Sigma_new = (np.eye(len(mu)) - K) @ Sigma
    return mu_new, Sigma_new

def plotpoints(allpoints, mu, plotLine=True, plot_in_meters=False):
    ''' Plots all estimates, the landmarks, and the associations between them '''
    if not allpoints:
        return  # Early exit if no points
    
    # Convert to numpy arrays for efficiency
    points = np.array([p[0] for p in allpoints])  # shape (n, 2): [lat, lon]
    idxs = np.array([p[1] for p in allpoints], dtype=int)
    mu_arr = np.array(mu)  # shape (m, 2): [lat, lon]
    
    # Compute frequency using bincount for O(n) efficiency
    frequency = np.bincount(idxs, minlength=len(mu_arr))
    
    if plot_in_meters:
        # Convert to meters relative to min_lat, min_lon, using average lat for lon scaling
        lats = points[:, 0]
        lons = points[:, 1]
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)
        avg_lat = (min_lat + max_lat) / 2
        METERS_PER_DEG_LON = METERS_PER_DEG_LAT * np.cos(np.deg2rad(avg_lat))
        
        # Compute meter coordinates: x (easting from west), y (northing from south)
        x_plot = (lons - min_lon) * METERS_PER_DEG_LON
        y_plot = (lats - min_lat) * METERS_PER_DEG_LAT
        mu_x = (mu_arr[:, 1] - min_lon) * METERS_PER_DEG_LON
        mu_y = (mu_arr[:, 0] - min_lat) * METERS_PER_DEG_LAT
        
        # Plot all photo estimates in one scatter call
        plt.scatter(x_plot, y_plot, color='red', s=10, alpha=1.0, label='photo estimate')
        
        # Plot associations as a LineCollection for efficiency with many lines
        if plotLine:
            starts = np.column_stack((x_plot, y_plot))
            ends = np.column_stack((mu_x[idxs], mu_y[idxs]))
            segments = np.stack((starts, ends), axis=1)  # shape (n, 2, 2)
            lc = LineCollection(segments, colors='gray', linewidths=0.5, alpha=1.0, label='association')
            plt.gca().add_collection(lc)
        
        # Plot landmarks with colors based on frequency in one scatter
        land_colors = np.where(frequency > 1, 'blue', 'orange')
        plt.scatter(mu_x, mu_y, s=10, c=land_colors, marker='x', label='landmark')
        
        plt.xlabel('Easting from West (meters)')
        plt.ylabel('Northing from South (meters)')
    else:
        # Original degree-based plotting
        # Plot all photo estimates in one scatter call
        plt.scatter(points[:, 1], points[:, 0], color='red', s=10, alpha=1.0, label='photo estimate')
        
        # Plot associations as a LineCollection for efficiency with many lines
        if plotLine:
            starts = points[:, [1, 0]]  # [lon, lat]
            ends = mu_arr[idxs][:, [1, 0]]  # Select rows first, then columns
            segments = np.stack((starts, ends), axis=1)  # shape (n, 2, 2)
            lc = LineCollection(segments, colors='gray', linewidths=0.5, alpha=1.0, label='association')
            plt.gca().add_collection(lc)
        
        # Plot landmarks with colors based on frequency in one scatter
        land_colors = np.where(frequency > 1, 'blue', 'orange')
        plt.scatter(mu_arr[:, 1], mu_arr[:, 0],c=land_colors, s=10, marker='x', label='landmark')
        
        plt.xlabel('Longitude (degrees)')
        plt.ylabel('Latitude (degrees)')
    
    # Legend proxies (unchanged)
    photo_proxy = plt.Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=2, label='photo estimate')
    landmark_proxy = plt.Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=3, label='landmark')
    assoc_proxy = plt.Line2D([0], [0], color='gray', linewidth=0.4, label='association')
    plt.legend(handles=[photo_proxy, landmark_proxy, assoc_proxy], loc='upper left')
    
    plt.axis('equal')

def associate_landmarks(input_data, MAH_threshold=8.0):
    mu = []  # List of landmark means (in lat/lon degrees)
    Sigma = []  # List of landmark covariances
    allpoints = []
    METERS_PER_DEG_LAT = 111319.9  # Approximate meters per degree latitude

    # Handle input flexibility: df or pre-parsed list of (points_np, unc)
    if isinstance(input_data, pd.DataFrame):
        # Parse from df (for standalone runs)
        df = input_data
        df['GPSparsed'] = df['GPSpoint'].apply(convertGPS)
        point_unc_list = []
        for index, row in df.iterrows():
            points = row['GPSparsed']
            if points.size > 0:  # Skip empty
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                point_unc_list.append((points, row['horizontal_accuracy']))
    elif isinstance(input_data, list):
        # Assume pre-parsed list from DA_verifier
        point_unc_list = input_data
    else:
        raise ValueError("Input must be a Pandas DataFrame or list of (points_np, unc) tuples")

    # Core association loop (efficient for large lists)
    for points, unc in point_unc_list:
        for point in points:  # Efficient: n small per entry
            lat = point[0]
            # Covariance in degrees^2 (scalar ops for speed)
            sigma_lat = unc / METERS_PER_DEG_LAT
            cos_lat = np.cos(np.deg2rad(lat))
            sigma_lon = unc / (METERS_PER_DEG_LAT * cos_lat)
            Sigma_bar = np.diag([sigma_lat**2, sigma_lon**2])
            
            association = associateData(point, mu, Sigma_bar, thresh=MAH_threshold)
            if association[0] == -1:
                allpoints.append([point, len(mu)])
                mu.append(point)
                Sigma.append(Sigma_bar)
            else:
                idx = association[0]
                mu[idx], Sigma[idx] = updatelandmark(mu[idx], Sigma[idx], point, Sigma_bar)
                allpoints.append([point, idx])
    return mu, allpoints

if __name__ == "__main__":
    start = time.time()

    csvpath = r"D:\CSVs\flowers_20251108_12pm.csv"  # Project CSV example

    # Load df from CSV (low_memory=False for large files)
    df = pd.read_csv(csvpath, low_memory=False)
    df = df[df['flower?'] == 1]  # Filter to flowers only (redundant for pre-filtered CSVs but robust)
    df = df.iloc[:1000]  # Subset for testing; remove for full run
    print(df.head())

    print('time to load csv is ', time.time() - start, 'seconds')

    mu, allpoints = associate_landmarks(df)

    print('Number of detected landmarks: ', len(mu))
    plotpoints(allpoints, mu, plot_in_meters=True)

    plt.show()
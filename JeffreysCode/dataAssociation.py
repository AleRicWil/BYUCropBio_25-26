import numpy as np
import pandas as pd
import time
import ast
import matplotlib.pyplot as plt
from nn import associateData

# plot gps points from csv

def convertGPS(GPSpoint):
    GPSpoint = ast.literal_eval(GPSpoint)
    GPSpoints = np.array([[float(x) for x in item.strip('[]').split()] for item in GPSpoint])
    return GPSpoints

def GPS2meters(GPSpoint):
    lat0, lon0 = 39.627024, -111.636855  # reference point
    # approximate conversion from lat/lon to meters
    lat, lon = GPSpoint
    latdiff = lat - lat0
    londiff = lon - lon0
    R = 6371000  # radius of Earth in meters
    x = R * np.deg2rad(londiff) * np.cos(np.deg2rad(latdiff))
    y = R * np.deg2rad(latdiff)
    return np.array([x, y])
    
def updatelandmark(mu, Sigma, z, R):
    K = Sigma @ np.linalg.inv(Sigma + R)
    mu_new = mu + K @ (z - mu)
    Sigma_new = (np.eye(len(mu)) - K) @ Sigma
    return mu_new, Sigma_new


def plotpoints(allpoints, mu, plotLine = True):
    ''' Plots all estimates, the landmarks, and the associations between them '''
    frequency = np.zeros(len(mu))
    # plot all estimate points
    for set in allpoints:
        point = set[0]
        idx = set[1]
        frequency[idx] += 1
        plt.scatter(point[0], point[1], color='red', s=1, label = 'photo estimate')
    # plot association line
        if plotLine:
            x_values = [point[0], mu[idx][0]]
            y_values = [point[1], mu[idx][1]]
            plt.plot(x_values, y_values, color='gray', linewidth=0.3, label ='association')

    # plot landmarks
    for i, point in enumerate(mu):
        col = 'blue' if frequency[i] > 1 else 'white'
        plt.scatter(point[0], point[1], color=col, s=2, label = 'landmark')
    
    # legend targets (proxy artists)
    photo_proxy = plt.Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=2, label='photo estimate')
    landmark_proxy = plt.Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=3, label='landmark')
    assoc_proxy = plt.Line2D([0], [0], color='gray', linewidth=0.4, label='association')
    plt.legend(handles=[photo_proxy, landmark_proxy, assoc_proxy], loc='upper left')
    plt.axis('equal')

def associate_landmarks(df):
    mu = []
    Sigma = []
    allpoints = []
    try:
        df['GPSparsed'] = df['GPSpoint'].apply(convertGPS)
    except:
        pass
    for index, row in df.iterrows():
        # for point in row['GPSparsed']:
            point = row['GPSparsed']
            # try:
            #     point = GPS2meters(point)
            #     print('converted point to meters')
            # except:
            #     pass
            unc = row['horizontal_accuracy']
            Sigma_bar = np.eye(2) * unc
            association = associateData(point, mu, Sigma_bar)
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

    csvpath  = "simulated_flower_data.csv" # define csv 

    
    # create df from csv
    df = pd.read_csv(csvpath)
    # df = df[df['flower?'] == 1]
    # df = df[df['sourceFolder'] == '10-31_2pm']
    # df = df[df['latitude'].notnull() & df['longitude'].notnull()]
    df = df.iloc[:1000]
    df['GPSparsed'] = df['GPSparsed'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    print(df.head())

    print('time to load new csv is ', time.time() - start, 'seconds')

    mu, allpoints = associate_landmarks(df)

    print('Number of detected landmarks: ', len(mu))
    plotpoints(allpoints, mu)

    plt.show()



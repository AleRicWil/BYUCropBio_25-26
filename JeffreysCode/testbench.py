import numpy as np
import pandas as pd
import time
import os
import ast
import camProj as cp
import matplotlib.pyplot as plt
from nn import associateDate as nn

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

# def plotpoints(df):
#     color_dict = {0: 'red', 1: 'orange', 2: 'green', 3: 'blue', 4: 'purple', 5: 'brown', 6: 'black'}
#     i = 0
#     col = 0
#     #plot gps points
    
#     for index, row in df.iterrows():
#         for point in convertGPS(row['GPSpoint']):
#             x, y = GPS2meters(point)
#             plt.scatter(x,y, color=color_dict.get(col), s=1)
#             i += 1
#             if i%10 == 0:
#                 col += 1
#                 if col > 6:
#                     col = 0
    
def updatelandmark(Sigma_old, Sigma_est, mu, estimate):
    K = Sigma_est @ np.linalg.inv(Sigma_old + Sigma_est)
    mu_new = mu + K @ (estimate - mu)
    Sigma_new = (np.eye(len(mu)) - K) @ Sigma_est
    return mu_new, Sigma_new
    S = H@Sigma_bar@H.T + Q
    K = Sigma_bar@H.T@np.linalg.inv(S)
    mu_bar += K@innovation
    mu_bar[2] = helpers.minimizedAngle(mu_bar[2])
    Sigma_bar = (np.eye(len(mu_bar)) - K@H[i])@Sigma_bar

def drawline(point1, point2):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, color='gray', linewidth=0.3, label ='association')

def plotpoints(allpoints, mu):
    frequency = np.zeros(len(mu))
    for set in allpoints:
        point = set[0]
        idx = set[1]
        frequency[idx] += 1
        plt.scatter(point[0], point[1], color='red', s=1, label = 'photo estimate')
        drawline(point, mu[idx])
    for i, point in enumerate(mu):
        if frequency[i] > 1:
            plt.scatter(point[0], point[1], color='blue', s=2, label = 'landmark')
        else:
            plt.scatter(point[0], point[1], color='white', s=2, label = 'landmark')


csvpath  = r"D:/working_data/fullgpscsv.csv"
start = time.time()
df = pd.read_csv(csvpath)


df = df[df['flower?'] == 1]
df = df[df['sourceFolder'] == '10-31_2pm']
df = df[df['latitude'].notnull() & df['longitude'].notnull()]

print('time to load new csv is ', time.time() - start, 'seconds')

# df = df.iloc[2:1000]
mu = []
Sigma = []
allpoints = []
for index, row in df.iterrows():
    for point in convertGPS(row['GPSpoint']):
        point = GPS2meters(point)
        # plt.scatter(point[0], point[1], color='red', s=0.5)
        unc = row['horizontal_accuracy']
        Sigma_bar = np.eye(2) * unc *2
        association = nn(point, mu, Sigma_bar)
        # print('association result: ', association)  
        if association[0] == -1:
            allpoints.append([point, len(mu)])
            mu.append(point)
            Sigma.append(Sigma_bar)
            
        else:
            idx = association[0]
            mu[idx], Sigma[idx] = updatelandmark(Sigma[idx], np.eye(2)*0.5, mu[idx], point)
            allpoints.append([point, idx])
    # print('associations: ', allpoints)


print('Number of landmarks: ', len(mu))
# for point in mu:
#     plt.scatter(point[0], point[1], color='blue', s=1)
plotpoints(allpoints, mu)
# plt.legend(['photo estimate', 'landmark', 'association'])
plt.show()



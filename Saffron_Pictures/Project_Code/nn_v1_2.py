import numpy as np
import pandas as pd

'''
A nearest neighbor algorithm that associates each flower estimate with its likeliest landmark and updates the flower's estimated position
'''

def Mahalanobis(landmark, estimate, Sigma):
    diff = estimate - landmark
    M = (diff).T @ np.linalg.inv(Sigma) @ (diff)
    
    return M

def associateData(estimate, mu, Sigma, thresh = 8):
    association = []
    # lat, lon = z
    if len(mu) == 0:
        association.append(-1)  # new landmark
        return association
    Dmin = Mahalanobis(mu[0], estimate, Sigma)
    nearest = 0
    num_lm = len(mu)
    for j in range(num_lm):
        D = Mahalanobis(mu[j], estimate, Sigma)
        if D < Dmin:
            Dmin = D
            nearest = j
    # print('closest landmark at index ', nearest, ' with distance ', D)

    if Dmin <= thresh:
        association.append(nearest)
        # update landmark position   
    else:
        association.append(-1)         
    return association

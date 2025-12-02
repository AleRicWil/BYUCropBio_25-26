import numpy as np
import pandas as pd

'''
A nearest neighbor algorithm that associates each flower estimate with its likeliest landmark and updates the flower's estimated position
'''

def Mahalanobis(landmark, estimate, Sigma):
    diff = estimate - landmark
    M = (diff).T @ np.linalg.inv(Sigma) @ (diff)
    
    return M

def associateDate(estimate, mu, Sigma, thresh = 0.5):
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

def updateLandmarks(estimate, Sigma):
    pass



# take one measurement (estimate)
# if its the first one, add it as a new landmark
# else, compute Mahalanobis distance to all existing landmarks
# if min distance < threshold, associate measurement with that landmark and update landmark position

    # '''
    # Args:
    #     estimate: np.array shape (N, 4) where each row is (x, y, landmark_x, landmark_y)
    #     Sigma: np.array shape (2, 2) covariance matrix of the measurement noise
    # Returns:
    #     updated_estimate: np.array shape (N, 4) with updated (x, y) positions
    # '''
    # updated_estimate = estimate.copy()
    # Sigma_inv = np.linalg.inv(Sigma)
    
    # for i in range(estimate.shape[0]):
    #     x, y, landmark_x, landmark_y = estimate[i]
    #     diff = np.array([x - landmark_x, y - landmark_y])
    #     mahalanobis_distance = diff.T @ Sigma_inv @ diff
        
    #     # Update position based on Mahalanobis distance
    #     if mahalanobis_distance < 5.991:  # Chi-square threshold for 95% confidence with 2 DOF
    #         updated_estimate[i, 0] = landmark_x
    #         updated_estimate[i, 1] = landmark_y
            
    # return updated_estimate
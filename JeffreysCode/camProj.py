import numpy as np
import cv2
import os
import transforms as tr
import matplotlib.pyplot as plt

R_c2w = tr.rotz(-np.pi/2) @ tr.rotx(-np.deg2rad(90+35))
R_w2c = tr.rotx(np.deg2rad(90+35)) @ tr.rotz(np.pi/2)


### -- Cam 1 -- ###
# SN 217222062474
# Intrinsics
res1 = [1920, 1080]
fx1 = 1366.24
fy1 = 1365.39
cx1 = 950.96
cy1 = 536.82

K1 = np.array([[fx1, 0, cx1],
                [0, fy1, cy1],
                [0, 0, 1]])
dist_coeffs1 = np.array([0,0,0,0,0])

# Transform
t1_w2c_inw = np.array([0.04, 0.60025, -1.143]).T # m
T1_c2w = tr.se3(R_c2w, t1_w2c_inw) # T_c2w @ p converts point from camera frame to world frame

### --- Cam 2 --- ###
# SN 319522065401
# Intrinsics
res2 = [1920, 1080]
fx2 = 1365.39
fy2 = 1363.63
cx2 = 945.1
cy2 = 532.91

K2 = np.array([[fx2, 0, cx2],
                [0, fy2, cy2],
                [0, 0, 1]])
dist_coeffs2 = np.array([0,0,0,0,0])

# Transform
t2_w2c_inw = np.array([0.04, 0.32085, -1.143]).T
T2_c2w = tr.se3(R_c2w, t2_w2c_inw) # T_c2w @ p converts point from camera frame to world frame

### --- Cam 3 --- ###
# SN 217222067750
# Intrinsics
res3 = [1920, 1080]
fx3, fy3 = 1366.94, 1364.94
cx3, cy3 = 974.27, 523.67

K3 = np.array([[fx3, 0, cx3],
                [0, fy3, cy3],
                [0, 0, 1]])
dist_coeffs3 = np.array([0,0,0,0,0])

# Transform
t3_w2c_inw = np.array([0.04, 0.0605, -1.143]).T
T3_c2w = tr.se3(R_c2w, t3_w2c_inw) # T_c2w @ p converts point from camera frame to world frame

# Define plane
P = np.array([0.0, 0.0, -1.3])        # point on plane
N = np.array([0.0, 0.0, 1.0])  

def convertGPS(basecoord, delta):
    lat, lon = basecoord
    north = delta[0]  # Y axis is north-south
    west = delta[1]   # X axis is east-west

    # Using WGS-84 ellipsoid approximations
    M_phi = 111132.92 - 559.82 * np.cos(2 * np.deg2rad(lat)) + 1.175 * np.cos(4 * np.deg2rad(lat)) - 0.0023 * np.cos(6 * np.deg2rad(lat))
    M_lambda = 111412.84 * np.cos(np.deg2rad(lat)) - 93.5 * np.cos(3 * np.deg2rad(lat)) + 0.118 * np.cos(5 * np.deg2rad(lat))

    lat += north / M_phi  # Approx meters per degree latitude
    lon -= west / M_lambda
    return np.array([lat, lon])

def plotray(start_point, direction, length, ax, color='blue'):
    end_point = start_point + direction * length
    ax.plot3D(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        [start_point[2], end_point[2]],
        color=color,
        linewidth=2
    )


def plotcameraview(ax, cam_name, col='blue'):
 
    if '1' in cam_name:
        cam = 'cam1'
    elif '2' in cam_name:
        cam = 'cam2'
    elif '3' in cam_name:
        cam = 'cam3'
    # Create 3D figure axes

    ax.set_xlim([-.25, 1.5])
    ax.set_ylim([-.5, 1.25])
    ax.set_zlim([-1.5, 0])

    res = [1920,1080]

    # Get corners of field of view
    corners = np.array([[0,         0,     1],
                        [res[0],    0,     1],
                        [res[0],  res[1],  1],
                        [0,       res[1],  1]]).T  # 3x4
    
    cam_in_w, points3D, t = project3D(cam, corners)
    for point in points3D:
        ax.plot([cam_in_w[0], point[0]], [cam_in_w[1], point[1]], [cam_in_w[2], point[2]], color=col)

    ax.scatter(cam_in_w[0], cam_in_w[1], cam_in_w[2], color = col, marker = 'o', s=50) # Cameras as colored circles
    
    # # Plot antenna and cameras
    antenna_in_w = np.array([0,0,0,0]).T       # Antenna at origin
    ax.scatter(antenna_in_w[0], antenna_in_w[1], antenna_in_w[2], color='green', marker='^', s=100, label='Antenna') # Antenna as green triangle
        


def intersect_ray_plane(ray_origin, ray_dir, plane_point, plane_normal, epsilon=1e-8):
    """
    Computes the intersection of a ray and a plane.

    Args:
        ray_origin: np.array shape (3,) point where the ray starts
        ray_dir: np.array shape (3,) direction of the ray
        plane_point: np.array shape (3,) a point on the plane
        plane_normal: np.array shape (3,) normal vector of the plane
        epsilon: threshold to treat dot product as zero (parallel)

    Returns:
        intersection_point: np.array shape (3,) or None if no intersection
        t: parameter along the ray, or None if no intersection
    """
    ray_origin = np.asarray(ray_origin)
    ray_dir = np.asarray(ray_dir)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)

    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < epsilon:
        print('Ray is parallel to plane')
        return None, None

    t = np.dot(plane_point - ray_origin, plane_normal) / denom

    if t < 0:
        print('Intersection occurs "behind" the ray origin')
        return None, None

    intersection = ray_origin + t * ray_dir
    return intersection, t

def project3D(cam_name, points):    
    if '1' in cam_name:
        K = K1
        T_c2w = T1_c2w
    elif '2' in cam_name:
        K = K2
        T_c2w = T2_c2w
    elif '3' in cam_name:
        K = K3
        T_c2w = T3_c2w
    
    
    points3D = []
    ts = []
    cam_in_w = T_c2w @ np.array([0,0,0,1]).T # Camera 1 position in world frame
    print(f'camera {cam_name} in world frame: ', cam_in_w[:3])
    X_in_c = np.linalg.inv(K) @ points  # Ray in 3D in camera frame
    X_in_w = T_c2w[:3,:3] @ X_in_c + T_c2w[:3,3:4]  # Ray in 3D in world frame
    for i in range(X_in_w.shape[1]):
        points, t = intersect_ray_plane(cam_in_w[:3], X_in_w[:,i] - cam_in_w[:3], P, N)
        points3D.append(points)
        ts.append(t)

    return cam_in_w, points3D, t


def pxl2pnt3D(cam_name, pxl):
    if '1' in cam_name:
        K = K1
        T_c2w = T1_c2w
    elif '2' in cam_name:
        K = K2
        T_c2w = T2_c2w
    elif '3' in cam_name:
        K = K3
        T_c2w = T3_c2w
    
    pxl_h = np.array([pxl[0], pxl[1], 1]).T  # Homogeneous pixel coordinates
    X_in_c = np.linalg.inv(K) @ pxl_h  # Ray in 3D in camera frame
    X_in_w = T_c2w[:3,:3] @ X_in_c + T_c2w[:3,3:4].reshape(-1)  # Ray in 3D in world frame
    cam_in_w = T_c2w @ np.array([0,0,0,1]).T # Camera position in world frame

    point3D, t = intersect_ray_plane(cam_in_w[:3], X_in_w - cam_in_w[:3], P, N)
    return point3D
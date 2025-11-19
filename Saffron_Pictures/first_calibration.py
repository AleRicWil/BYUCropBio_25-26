import numpy as np
import json
from pathlib import Path

def load_intersections(json_path):
    """
    Loads the snapped intersections from the JSON file.
    Returns a list of dictionaries with 'name' and 'point' [x, y].
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['intersections']

def get_world_point(name, origin_char='A', origin_row=0, step=2.0):
    """
    Computes the world coordinates (X, Y) in cm for a given grid name like 'B-21'.
    Assumes columns are characters starting from origin_char (e.g., 'A' = 0),
    rows are integers starting from origin_row (e.g., 0 = 0 cm).
    X increases with column index, Y increases with row index.
    """
    col_char, row_str = name.split('-')
    col_idx = ord(col_char) - ord(origin_char)
    row_idx = int(row_str) - origin_row
    X = col_idx * step
    Y = row_idx * step
    return np.array([X, Y])

def estimate_homography(image_points, world_points):
    """
    Estimates the 3x3 homography matrix H such that image_pt ~ H @ [world_x, world_y, 1].
    Uses least-squares solution via SVD.
    Requires at least 4 point correspondences.
    Returns H (normalized so H[2,2] = 1).
    """
    assert len(image_points) == len(world_points) >= 4, "Need at least 4 points"
    
    A = []
    for (u, v), (X, Y) in zip(image_points, world_points):
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-6:
        H /= H[2, 2]
    else:
        raise ValueError("Homography estimation failed (singular matrix)")
    
    return H

def compute_reprojection_error(image_points, world_points, H):
    """
    Computes the mean reprojection error in pixels.
    Useful for verifying the quality of the homography.
    """
    errors = []
    for (u, v), (X, Y) in zip(image_points, world_points):
        world_hom = np.array([X, Y, 1])
        projected = H @ world_hom
        projected /= projected[2] if abs(projected[2]) > 1e-6 else 1
        error = np.hypot(projected[0] - u, projected[1] - v)
        errors.append(error)
    return np.mean(errors)

def compute_projection_matrix(json_path, origin_char='A', origin_row=0, step=2.0):
    """
    Main function to compute the homography (projection matrix) for a single image.
    - Loads intersections from JSON.
    - Maps names to world coordinates.
    - Estimates homography H (world to image).
    - Computes and prints reprojection error.
    - Returns H and the base name for saving.
    """
    intersections = load_intersections(json_path)
    image_points = np.array([inter['point'] for inter in intersections], dtype=float)
    world_points = np.array([get_world_point(inter['name'], origin_char, origin_row, step) for inter in intersections], dtype=float)
    
    H = estimate_homography(image_points, world_points)
    
    error = compute_reprojection_error(image_points, world_points, H)
    print(f"Reprojection error for {json_path}: {error:.2f} pixels")
    
    base_name = Path(json_path).stem.replace('_intersections_corrected', '')
    return H, base_name

# Example usage for left, middle, right images
# Adjust paths and parameters (origin_char, origin_row) based on your grid labeling
left_json = 'left_intersections_corrected.json'
middle_json = 'middle_intersections_corrected.json'
right_json = 'right_intersections_corrected.json'

H_left, base_left = compute_projection_matrix(left_json)
H_middle, base_middle = compute_projection_matrix(middle_json)
H_right, base_right = compute_projection_matrix(right_json)

# Save the matrices (e.g., for later use in warping other images)
np.save(f'{base_left}_homography.npy', H_left)
np.save(f'{base_middle}_homography.npy', H_middle)
np.save(f'{base_right}_homography.npy', H_right)

print("Homography matrices saved. To apply to other images, load H and use matrix multiplication for point projection or implement image warping.")
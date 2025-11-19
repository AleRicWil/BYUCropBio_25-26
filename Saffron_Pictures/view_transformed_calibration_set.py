import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import map_coordinates
from pathlib import Path
import json
import cv2

def load_homography(npy_path):
    """Load the homography matrix from .npy file."""
    return np.load(npy_path)

def load_intersections(json_path):
    """Load intersections to determine world bounds."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['intersections']

def get_world_bounds(intersections, step=2.0, margin=2.0):
    """Compute min/max world X,Y from intersections."""
    min_col = min(ord(inter['name'].split('-')[0]) for inter in intersections)
    max_col = max(ord(inter['name'].split('-')[0]) for inter in intersections)
    min_row = min(int(inter['name'].split('-')[1]) for inter in intersections)
    max_row = max(int(inter['name'].split('-')[1]) for inter in intersections)
    min_X = (min_col - ord('A')) * step - margin
    max_X = (max_col - ord('A') + 1) * step + margin
    min_Y = (min_row - 0) * step - margin  # Assuming origin_row=0
    max_Y = (max_row - 0 + 1) * step + margin
    return min_X, max_X, min_Y, max_Y

def warp_image(image, H, output_shape, min_X, min_Y, scale=50.0):  # scale: pixels per cm
    """Warp the image to top-down view using the homography."""
    height, width = output_shape
    warped = np.zeros((height, width, 3), dtype=image.dtype)
    
    # Create meshgrid for world coordinates
    X, Y = np.meshgrid(np.linspace(min_X, min_X + width / scale, width),
                       np.linspace(min_Y, min_Y + height / scale, height))
    
    # Stack to homogeneous world points
    world_pts = np.stack([X.ravel(), Y.ravel(), np.ones_like(X.ravel())], axis=0)
    
    # Project to image coordinates: image_pts = H @ world_pts
    image_pts = H @ world_pts
    image_pts /= image_pts[2, :] + 1e-6  # Normalize
    
    u = image_pts[0, :].reshape(height, width)
    v = image_pts[1, :].reshape(height, width)
    
    # Sample from original image using bilinear interpolation
    for c in range(3):
        warped[:, :, c] = map_coordinates(image[:, :, c], [v, u], order=1, mode='constant', cval=0)
    
    return warped

# Paths (adjust as needed)
views = ['left', 'middle', 'right']
original_images = {}
warped_images = {}
bounds = {}

# Load originals and compute warped
for view in views:
    image_path = f'{view}.jpg'
    json_path = f'{view}_intersections_corrected.json'  # Or _corrected.json if fixed
    homography_path = f'{view}_homography.npy'
    
    # Load original
    original_images[view] = mpimg.imread(image_path)
    
    # Load homography
    H = load_homography(homography_path)
    
    # Get bounds
    intersections = load_intersections(json_path)
    min_X, max_X, min_Y, max_Y = get_world_bounds(intersections)
    width_cm = max_X - min_X
    height_cm = max_Y - min_Y
    scale = 50.0  # Pixels per cm, adjust for resolution
    output_shape = (int(height_cm * scale), int(width_cm * scale))
    bounds[view] = (min_X, max_X, min_Y, max_Y)
    
    # Warp
    warped_images[view] = warp_image(original_images[view], H, output_shape, min_X, min_Y, scale)

# warped_images = refine_homographies(warped_images, bounds, scale)

# Compute global bounds across all views
global_min_X = min(bounds[view][0] for view in views)
global_max_X = max(bounds[view][1] for view in views)
global_min_Y = min(bounds[view][2] for view in views)
global_max_Y = max(bounds[view][3] for view in views)
global_width_cm = global_max_X - global_min_X
global_height_cm = global_max_Y - global_min_Y
global_shape = (int(global_height_cm * scale), int(global_width_cm * scale))

# Create composite image (initialize to black)
composite = np.zeros((*global_shape, 3), dtype=np.uint8)
count_map = np.zeros(global_shape, dtype=int)  # For averaging

# Overlay each warped image
for view in views:
    warped = warped_images[view]
    min_X, _, min_Y, _ = bounds[view]
    offset_x = int((min_X - global_min_X) * scale)
    offset_y = int((min_Y - global_min_Y) * scale)
    h, w = warped.shape[:2]
    
    # Add to composite (average where overlap)
    composite[offset_y:offset_y + h, offset_x:offset_x + w] += warped
    count_map[offset_y:offset_y + h, offset_x:offset_x + w] += 1

# Average
composite = composite.astype(np.uint32)
divisor = np.maximum(count_map, 1)[:, :, np.newaxis].astype(np.uint32)
composite //= divisor
composite = composite.astype(np.uint8)

# Display originals in one window
fig_orig, axs_orig = plt.subplots(1, 3, figsize=(15, 5))
for i, view in enumerate(views):
    axs_orig[i].imshow(original_images[view])
    axs_orig[i].set_title(f'Original {view.capitalize()}')
    axs_orig[i].axis('off')

# Display warped in another window
fig_warped, axs_warped = plt.subplots(1, 3, figsize=(15, 5))
for i, view in enumerate(views):
    axs_warped[i].imshow(warped_images[view])
    axs_warped[i].set_title(f'Warped {view.capitalize()}')
    axs_warped[i].axis('off')

# Display overlapped composite in a third window
fig_composite, ax_composite = plt.subplots(figsize=(10, 10))
ax_composite.imshow(composite)
ax_composite.set_title('Overlapped Global Grid')
ax_composite.axis('off')

plt.show()
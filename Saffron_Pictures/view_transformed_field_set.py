import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import map_coordinates
from pathlib import Path

def load_homography(npy_path):
    """Load the homography matrix from .npy file."""
    return np.load(npy_path)

def get_world_bounds_from_homography(H, image_shape, scale=50.0):
    """
    Estimates world bounds by projecting image corners using the homography.
    Assumes a reasonable grid extent; refine if needed for specific setups.
    """
    height, width = image_shape[:2]
    corners = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]]).T
    world_corners = np.linalg.inv(H) @ corners  # Inverse to map image to world
    world_corners /= world_corners[2, :]
    min_X, max_X = np.min(world_corners[0, :]), np.max(world_corners[0, :])
    min_Y, max_Y = np.min(world_corners[1, :]), np.max(world_corners[1, :])
    return min_X, max_X, min_Y, max_Y

def warp_image(image, H, output_shape, min_X, min_Y, scale=50.0):
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

# Paths (adjust as needed for the new trio of images)
views = ['left', 'middle', 'right']
image_paths = [r"C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\RS1\RS1_color_20251028_042339_376551Z.jpg",
               r"C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\RS2\RS2_color_20251028_042339_386636Z.jpg",
               r"C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\RS3\RS3_color_20251028_042339_366756Z.jpg"]
images = {}
homographies = {}
bounds = {}
scale = 50.0  # Pixels per cm, consistent with calibration

# Load images and homographies
for view, image_path in zip(views, image_paths):
    homography_path = f'{view}_homography.npy'  # From calibration
    
    images[view] = mpimg.imread(image_path)
    homographies[view] = load_homography(homography_path)
    
    # Estimate bounds from homography and image shape
    min_X, max_X, min_Y, max_Y = get_world_bounds_from_homography(homographies[view], images[view].shape)
    bounds[view] = (min_X, max_X, min_Y, max_Y)

# Compute global bounds across all views
global_min_X = min(bounds[view][0] for view in views)
global_max_X = max(bounds[view][1] for view in views)
global_min_Y = min(bounds[view][2] for view in views)
global_max_Y = max(bounds[view][3] for view in views)
global_width_cm = global_max_X - global_min_X
global_height_cm = global_max_Y - global_min_Y
global_shape = (int(global_height_cm * scale), int(global_width_cm * scale))

# Warp each image to its own bounds
warped_images = {}
for view in views:
    min_X, max_X, min_Y, max_Y = bounds[view]
    width_cm = max_X - min_X
    height_cm = max_Y - min_Y
    output_shape = (int(height_cm * scale), int(width_cm * scale))
    warped_images[view] = warp_image(images[view], homographies[view], output_shape, min_X, min_Y, scale)

# Create composite image (initialize to black)
composite = np.zeros((*global_shape, 3), dtype=np.uint32)  # Use uint32 to avoid overflow
count_map = np.zeros(global_shape, dtype=int)  # For averaging

# Overlay each warped image
for view in views:
    warped = warped_images[view]
    min_X, _, min_Y, _ = bounds[view]
    offset_x = int((min_X - global_min_X) * scale)
    offset_y = int((min_Y - global_min_Y) * scale)
    h, w = warped.shape[:2]
    
    # Add to composite
    composite[offset_y:offset_y + h, offset_x:offset_x + w] += warped.astype(np.uint32)
    count_map[offset_y:offset_y + h, offset_x:offset_x + w] += 1

# Average where overlapped
mask = count_map > 0
divisor = np.maximum(count_map, 1)[:, :, np.newaxis].astype(np.uint32)
composite[mask] //= divisor[mask]
composite = composite.astype(np.uint8)

# Display the overlapped composite
fig_composite, ax_composite = plt.subplots(figsize=(10, 10))
ax_composite.imshow(composite)
ax_composite.set_title('Overlapped Global View of Subject Images')
ax_composite.axis('off')
plt.show()
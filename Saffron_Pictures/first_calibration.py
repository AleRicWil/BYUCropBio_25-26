import cv2
import numpy as np

# Step 1: Define constants from your setup
# Grid size from your PDF (e.g., num_cols x num_rows squares, but we use inner corners: (num_cols-1) x (num_rows-1))
# Example: For 100cm x 60cm at 2cm squares -> 50x30 squares -> 49x29 corners (adjust based on your generate_checkerboard_pdf call)
pattern_size = (49, 29)  # (width-1, height-1) in corners; width=cols (numbers), height=rows (letters)
square_size_cm = 2.0  # From your pattern

# Step 2: Load intrinsics from datasheet (example placeholders; replace with yours)
# K: 3x3 intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
# dist_coeffs: [k1, k2, p1, p2, k3] for radial/tangential distortion
K_left = np.array([[fx_left, 0, cx_left], [0, fy_left, cy_left], [0, 0, 1]], dtype=np.float32)
dist_left = np.array([k1_left, k2_left, p1_left, p2_left, k3_left], dtype=np.float32)

K_middle = np.array([[fx_middle, 0, cx_middle], [0, fy_middle, cy_middle], [0, 0, 1]], dtype=np.float32)
dist_middle = np.array([k1_middle, k2_middle, p1_middle, p2_middle, k3_middle], dtype=np.float32)

K_right = np.array([[fx_right, 0, cx_right], [0, fy_right, cy_right], [0, 0, 1]], dtype=np.float32)
dist_right = np.array([k1_right, k2_right, p1_right, p2_right, k3_right], dtype=np.float32)

# Image file paths (from your capture)
img_left_path = 'left.jpg'
img_middle_path = 'middle.jpg'
img_right_path = 'right.jpg'

# Step 3: Generate 3D object points (world coordinates, in cm)
# Corners: from (0,0,0) bottom-left, x along columns (numbers), y along rows (letters), z=0 plane
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_cm
# Key concept: This assumes row-major ordering matching detection; labels help verify visually

# Step 4: Function to detect corners and compute pose for one camera
def compute_pose(img_path, K, dist):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find corners (sub-pixel refinement for accuracy)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Undistort corners if distortion is significant
        corners_undist = cv2.undistortPoints(corners, K, dist, P=K)
        corners_undist = np.squeeze(corners_undist)  # To 2D
        
        # Solve PnP for extrinsics (R, t)
        # Key concept: Uses objp (3D) and corners (2D) to estimate pose via iterative method
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if ret:
            R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
            return R, tvec
        else:
            print(f"PnP failed for {img_path}")
            return None, None
    else:
        print(f"Checkerboard not found in {img_path}")
        return None, None

# Step 5: Compute poses for each camera
R_left, t_left = compute_pose(img_left_path, K_left, dist_left)
R_middle, t_middle = compute_pose(img_middle_path, K_middle, dist_middle)
R_right, t_right = compute_pose(img_right_path, K_right, dist_right)

# Step 6: Compute relative extrinsics (e.g., left relative to middle)
# Key concept: Relative pose = target_pose * source_pose.inv()
# For stereo: Use these in triangulation or stereoRectify
if R_left is not None and R_middle is not None:
    R_left_to_middle = np.dot(R_middle, np.linalg.inv(R_left))
    t_left_to_middle = t_middle - np.dot(R_left_to_middle, t_left)
    print("Left to Middle Relative R:\n", R_left_to_middle)
    print("Left to Middle Relative t:\n", t_left_to_middle)

if R_middle is not None and R_right is not None:
    R_middle_to_right = np.dot(R_right, np.linalg.inv(R_middle))
    t_middle_to_right = t_right - np.dot(R_middle_to_right, t_middle)
    print("Middle to Right Relative R:\n", R_middle_to_right)
    print("Middle to Right Relative t:\n", t_middle_to_right)

# Optional: Visualize detections (comment out to run)
# cv2.drawChessboardCorners(img_left, pattern_size, corners, ret)  # Repeat for each

# Step 7: Save extrinsics (e.g., for later use in 3D reconstruction)
np.savez('extrinsics.npz', R_left=R_left, t_left=t_left, R_middle=R_middle, t_middle=t_middle,
         R_right=R_right, t_right=t_right)
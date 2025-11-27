# warp_and_stitch_fast.py
# Ultra-fast batch warping for fixed 3-camera rig on flat ground
# Optimizations:
# - Precompute global bounds from homography-projected corners → no JSON needed, auto-fits canvas
# - User-configurable pixels per meter for output resolution control
# - Proper sum-based averaging for composite, excluding black (zero) regions to avoid darkening
# - Parallel processing of entire frames (warp + stitch + save) for max throughput
# - Stricter non-zero mask with threshold to ignore near-black noise
# - Sequential grouping of images: first RS1 + first RS2 + first RS3, etc., after sorting each by timestamp
# - Handle mismatched list lengths by taking min length
# - Optional faster interpolation (NEAREST) for speed vs. quality tradeoff
# - Adjusted for RS1 (left), RS2 (middle), RS3 (right) folder structure
# - Loads example images from calibrate subfolders for shape and bounds precomputation
# - Globs images from RS1/RS2/RS3 subfolders in the input batch folder
# - Note: GPS data from captures.csv is not yet integrated; this script focuses on image warping/stitching.
#   For flower mapping, we'll add detection (e.g., color thresholding or ML) and GPS association in a future extension.

import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial  # For partial function application in multiprocessing
import re  # For timestamp parsing in filenames

# ================== CONFIGURATION ==================
# Set desired output resolution (pixels per meter)
PIXELS_PER_METER = 2000.0  # Example: 100 px/m ≈ 1 cm/pixel; adjust for quality vs. speed/memory
MARGIN_M = 2.0            # Extra margin around the field in meters to avoid clipping edges
INTERPOLATION = cv2.INTER_LINEAR  # Or cv2.INTER_NEAREST for ~2x faster but lower quality
NON_ZERO_THRESHOLD = 30   # Sum RGB > this to count as non-black (avoids including dark noise in average)

# Camera views mapped to subfolders: RS1=left, RS2=middle, RS3=right
VIEWS = ['RS1', 'RS2', 'RS3']

# Homography source (assumes files named RS1_homography.npy etc.; adjust if using left/middle/right)
H_SOURCE = 'npy'  # or 'txt' for Transformations.txt

# Pattern for timestamp in filename (based on your sample: RS1_color_20251028_042339_376551Z.jpg)
TS_PATTERN = re.compile(r'(\d{8}_\d{6}_\d{6}Z)')  # e.g., 20251028_042339_376551Z

# ===================================================

def load_homography(view: str):
    """
    Loads the homography matrix for the given view from either .npy or .txt source.
    Assumes .npy files are named like RS1_homography.npy.
    """
    if H_SOURCE == 'npy':
        path = Path(f"{view}_homography.npy")
        H = np.load(path)
    else:
        # Parse Transformations.txt (assuming Cam1=RS1/left, etc.)
        with open("Saffron_Pictures/Transformations.txt") as f:
            lines = f.readlines()
        cam_map = {"RS1": "Cam1", "RS2": "Cam2", "RS3": "Cam3"}
        cam = cam_map[view]
        start = next(i for i, l in enumerate(lines) if cam in l) + 1
        M = np.zeros((4, 4))
        for i in range(4):
            clean = lines[start + i].strip('[] \n')
            M[i] = [float(x) for x in clean.split()]
        H = M[:3, [0, 1, 3]]  # Extract 3x3 homography from 4x4 transformation
    print(H)
    return H.astype(np.float32)

def compute_bounds(H, img_shape, margin_m=0.0):
    """
    Computes the world bounds in meters by back-projecting the image corners using the inverse homography.
    Adds optional margin to avoid clipping.
    """
    h, w = img_shape[:2]
    corners_img = np.float32([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]]).T
    H_inv = np.linalg.inv(H)
    corners_world = H_inv @ corners_img
    corners_world /= corners_world[2, :] + 1e-6
    min_X, max_X = corners_world[0, :].min() - margin_m, corners_world[0, :].max() + margin_m
    min_Y, max_Y = corners_world[1, :].min() - margin_m, corners_world[1, :].max() + margin_m
    return min_X, max_X, min_Y, max_Y

def precompute(calibrate_folder: Path):
    """
    Precomputes homographies, per-view bounds, per-view output shapes, and remap maps.
    No global canvas; each view gets its own output shape for side-by-side positioning.
    """
    print("Loading homographies and computing per-view bounds...")
    Hs = {}
    bounds = {}
    out_shapes = {}
    maps = {}
    example_shapes = {}

    for view in VIEWS:
        # Load example image from calibrate/RS1/*.jpg (take the first one)
        view_dir = calibrate_folder / view
        example_paths = list(view_dir.glob('*.jpg'))
        if not example_paths:
            raise FileNotFoundError(f"No images found in {view_dir}")
        example_path = example_paths[0]  # Use first image for shape
        img = cv2.imread(str(example_path))
        if img is None:
            raise ValueError(f"Failed to load example image: {example_path}")
        example_shapes[view] = img.shape

        # Load H (world → image)
        Hs[view] = load_homography(view)

        # Compute per-view bounds with full margin
        min_X, max_X, min_Y, max_Y = compute_bounds(Hs[view], img.shape, MARGIN_M)
        bounds[view] = (min_X, max_X, min_Y, max_Y)

        # Compute per-view output shape (convert cm to m)
        width_cm = max_X - min_X
        height_cm = max_Y - min_Y
        width_m = width_cm / 100.0
        height_m = height_cm / 100.0
        out_w = int(width_m * PIXELS_PER_METER) + 1
        out_h = int(height_m * PIXELS_PER_METER) + 1
        out_shapes[view] = (out_h, out_w)
        print(f"{view}: {width_m:.1f}m x {height_m:.1f}m → output {out_w}x{out_h} px")

        # Precompute remap maps for this view's local coordinates
        world_X, world_Y = np.meshgrid(np.linspace(min_X, max_X, out_w),
                                       np.linspace(min_Y, max_Y, out_h))
        world_pts = np.stack([world_X.ravel(), world_Y.ravel(), np.ones_like(world_X.ravel())], axis=0)
        H = Hs[view]
        image_pts = H @ world_pts
        image_pts /= image_pts[2, :] + 1e-6
        map_x = image_pts[0, :].reshape(out_h, out_w).astype(np.float32)
        map_y = image_pts[1, :].reshape(out_h, out_w).astype(np.float32)
        maps[view] = (map_x, map_y)
        print(f"  {view:6s} → map created")

    return maps, out_shapes, bounds, example_shapes

def process_frame(triplet, maps, out_shapes, example_shapes, flat_dir, original_dir):
    """
    Processes a single frame triplet: warps each to its own canvas, then concatenates left to right (RS1 | RS2 | RS3).
    Handles partial triplets by leaving blank (black) panels for missing views.
    Also creates an original unwarped stacked image.
    """
    rs1_path, rs2_path, rs3_path = triplet
    paths = [p for p in [rs1_path, rs2_path, rs3_path] if p]
    frame_idx = "unknown"
    if paths:
        match = TS_PATTERN.search(str(paths[0]))
        if match:
            frame_idx = match.group(1)

    # Warp each view to its own canvas for flat (warped) version
    warps = []
    max_h_warped = max(out_shapes[view][0] for view in VIEWS)  # Align heights by padding
    # Also prepare originals
    originals = []
    orig_heights = []
    for view, path in zip(VIEWS, [rs1_path, rs2_path, rs3_path]):
        if path is None or not path.exists():
            # Blank for warped
            h, w = out_shapes[view]
            warped = np.zeros((h, w, 3), dtype=np.uint8)
            # Blank for original
            h_orig, w_orig = example_shapes[view][:2]
            img_orig = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(path))
            if img is None:
                # Blank for warped
                h, w = out_shapes[view]
                warped = np.zeros((h, w, 3), dtype=np.uint8)
                # Blank for original
                h_orig, w_orig = example_shapes[view][:2]
                img_orig = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            else:
                # For warped
                map_x, map_y = maps[view]
                warped = cv2.remap(img, map_x, map_y, INTERPOLATION,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                # For original
                img_orig = img
        # Pad warped to max height if needed
        if warped.shape[0] < max_h_warped:
            pad_top = (max_h_warped - warped.shape[0]) // 2
            pad_bottom = max_h_warped - warped.shape[0] - pad_top
            warped = cv2.copyMakeBorder(warped, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        warps.append(warped)
        # Collect original
        originals.append(img_orig)
        orig_heights.append(img_orig.shape[0])

    # Concatenate warped (flat)
    flat_composite = np.hstack(warps)

    # Pad originals to max orig height
    max_h_orig = max(orig_heights)
    for i in range(len(originals)):
        if originals[i].shape[0] < max_h_orig:
            pad_top = (max_h_orig - originals[i].shape[0]) // 2
            pad_bottom = max_h_orig - originals[i].shape[0] - pad_top
            originals[i] = cv2.copyMakeBorder(originals[i], pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

    # Concatenate originals
    original_composite = np.hstack(originals)

    # Save flat (warped)
    flat_path = Path(flat_dir) / f"stitched_{frame_idx}_flat.jpg"
    cv2.imwrite(str(flat_path), flat_composite)
    print(f"Saved {flat_path}")

    # Save original
    original_path = Path(original_dir) / f"stitched_{frame_idx}_original.jpg"
    cv2.imwrite(str(original_path), original_composite)
    print(f"Saved {original_path}")

def extract_ts(path):
    """
    Extracts timestamp from path for sorting.
    Returns '0' if no match to allow sorting.
    """
    match = TS_PATTERN.search(str(path))
    return match.group(1) if match else "0"

def main():
    """
    Main function: parses arguments, precomputes maps and bounds using calibrate folder,
    collects and sorts images from RS1/RS2/RS3 subfolders in the input batch,
    groups them sequentially by index after sorting (chronological order),
    and processes frames in parallel.
    """
    global PIXELS_PER_METER  # Declare global at the top before any use or assignment

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="C:\\Users\\Alex R. Williams\\Pictures\\Saffron Data\\calibrate", 
                        help="Batch folder with RS1/RS2/RS3 subfolders containing images")
    parser.add_argument('--calibrate', '-c', default="C:\\Users\\Alex R. Williams\\Pictures\\Saffron Data\\calibrate",
                        help="Path to calibrate folder for example images and bounds (default: hardcoded path)")
    # parser.add_argument('--output', '-o', default="stitched_output", help="Output folder for stitched images")  # No longer needed
    parser.add_argument('--threads', type=int, default=int(mp.cpu_count()*0.8), help="Number of parallel workers")
    parser.add_argument('--ppm', type=float, default=PIXELS_PER_METER, help="Pixels per meter for output resolution")
    args = parser.parse_args()

    # Override global config with arg if provided
    PIXELS_PER_METER = args.ppm

    # Dynamically set output folders (create if needed)
    input_path = Path(args.input)
    flat_dir = input_path / 'Combined_Flat'
    flat_dir.mkdir(exist_ok=True)
    original_dir = input_path / 'Combined_Original'
    original_dir.mkdir(exist_ok=True)

    # Precompute using calibrate folder
    calibrate_folder = Path(args.calibrate)
    maps, out_shapes, _, example_shapes = precompute(calibrate_folder)  # Ignore bounds return if not used

    print("Loading and processing captures.csv for triplet formation...")
    import pandas as pd  # Import here if not already at top; move to top if preferred

    csv_path = Path(args.input) / 'captures.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"captures.csv not found in {args.input}")

    df = pd.read_csv(csv_path)

    # Filter to relevant cameras and non-null timestamps
    df = df[df['camera'].isin(['RS1', 'RS2', 'RS3']) & df['timestamp'].notna()]

    # Function to get image path and check existence
    def get_image_path(row):
        camera = row['camera']
        timestamp = row['timestamp']
        path = Path(args.input) / camera / f"{camera}_color_{timestamp}.jpg"
        return path if path.exists() else None

    df['image_path'] = df.apply(get_image_path, axis=1)
    df = df[df['image_path'].notna()]  # Drop rows without matching images

    # Sort by timestamp (string sorts lexicographically for this format)
    df = df.sort_values('timestamp')

    # Find the first RS1 index
    rs1_mask = df['camera'] == 'RS1'
    if rs1_mask.any():
        start_idx = df[rs1_mask].index.min()
        sub_df = df.loc[start_idx:]
    else:
        sub_df = df  # Fallback to full DataFrame if no RS1
        print("Warning: No RS1 entries found; using full sequence as fallback")

    # Traverse to build variable-length triplets (lists of row dicts)
    triplets = []
    current_triplet = []
    prev_camera = None
    for _, row in sub_df.iterrows():
        camera = row['camera']
        if (camera == 'RS1') or (camera == prev_camera and camera in ['RS2', 'RS3']):
            if current_triplet:
                triplets.append(current_triplet)
            current_triplet = [row.to_dict()]  # Start new triplet with this row
        else:
            current_triplet.append(row.to_dict())  # Add to current triplet
        prev_camera = camera

    if current_triplet:
        triplets.append(current_triplet)

    # Convert to fixed [RS1_path, RS2_path, RS3_path] with Nones
    formatted_triplets = []
    cam_to_idx = {'RS1': 0, 'RS2': 1, 'RS3': 2}
    for trip in triplets:
        fmt = [None, None, None]
        for r in trip:
            idx = cam_to_idx[r['camera']]
            if fmt[idx] is not None:
                print(f"Warning: Duplicate {r['camera']} in triplet starting at timestamp {trip[0]['timestamp']}")
            fmt[idx] = r['image_path']
        if any(fmt):  # Skip if all None (shouldn't happen)
            formatted_triplets.append(fmt)

    # Sort formatted triplets by min timestamp (for processing order)
    def get_min_ts(trip):
        ts_list = []
        for p in trip:
            if p is not None:
                # Extract timestamp from path (e.g., ..._color_20251028_210522_886897Z.jpg)
                ts_str = p.stem.split('_color_')[1]
                ts_list.append(ts_str)
        return min(ts_list) if ts_list else 'zzzzzzzz_zzzzzz_zzzzzzZ'  # Fallback for sorting

    formatted_triplets.sort(key=get_min_ts)

    # Use as triplets for processing
    triplets = formatted_triplets
    print(f"Formed {len(triplets)} triplets from CSV data")

    # Process in parallel
    print(f"Processing with {args.threads} threads...")
    process_func = partial(process_frame, maps=maps, out_shapes=out_shapes, example_shapes=example_shapes, flat_dir=str(flat_dir), original_dir=str(original_dir))
    with mp.Pool(args.threads) as pool:
        pool.map(process_func, triplets)

    print(f"Done! Flat stitched frames in {flat_dir}, Original combined frames in {original_dir}")

if __name__ == '__main__':
    main()
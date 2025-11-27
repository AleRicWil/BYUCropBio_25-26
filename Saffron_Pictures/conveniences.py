import numpy as np
from pathlib import Path

def load_and_display_homography(npy_path):
    """
    Loads a homography matrix from a .npy file, prints it to the console,
    and saves it as a human-readable .txt file in the same directory.
    """
    if not Path(npy_path).exists():
        print(f"File not found: {npy_path}")
        return None
    
    H = np.load(npy_path)
    print(f"\nHomography matrix from {npy_path}:")
    print(H)
    
    # Save to .txt file
    txt_path = Path(npy_path).with_suffix('.txt')
    np.savetxt(txt_path, H, fmt='%.8e', header=f"Homography matrix from {npy_path}")
    print(f"Saved to: {txt_path}")
    
    return H

# List of homography files (adjust paths if needed)
views = ['left', 'middle', 'right']
homography_paths = [f'{view}_homography.npy' for view in views]

# Load and process each
for path in homography_paths:
    load_and_display_homography(path)
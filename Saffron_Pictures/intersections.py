import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.ndimage import sobel, gaussian_filter, maximum_filter
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression

# Load the JSON data (replace with your actual JSON file path)
json_path = 'right_lines.json'
with open(json_path, 'r') as f:
    data = json.load(f)

image_path = data['image']
horizontal_data = data['horizontal_lines']
vertical_data = data['vertical_lines']

# Load the image
img = mpimg.imread(image_path)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title('Line Viewer')

# Variables for panning
panning = False
start_x = None
start_y = None
orig_xlim = None
orig_ylim = None

# Helper function to compute infinite line intersection
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:  # Parallel or coincident
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    s = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)

# Function to calculate intersections
def calculate_intersections(horizontal_data, vertical_data):
    height, width = img.shape[:2]
    intersections = []
    for h in horizontal_data:
        (h_p1, h_p2) = h['points']
        h_label = h['label']
        for v in vertical_data:
            (v_p1, v_p2) = v['points']
            v_label = v['label']
            inter_point = line_intersection(h_p1, h_p2, v_p1, v_p2)
            if inter_point:
                ix, iy = inter_point
                if 0 <= ix <= width and 0 <= iy <= height:
                    name = f"{v_label}-{h_label}"
                    intersections.append({'name': name, 'point': inter_point})
    # Sort alpha-numerically: by Unicode order of char, then numerically by number
    intersections.sort(key=lambda i: (ord(i['name'].split('-')[0][0]), int(i['name'].split('-')[1])))
    return intersections

def subpixel_peak(pos, vals):
    """
    Simple parabolic sub-pixel refinement for a 1D peak.
    Assumes the input is three consecutive values: [left, center, right].
    Returns the fractional offset from the center pixel.
    """
    if len(vals) != 3:
        return 0
    left, center, right = vals
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-6:          # Almost flat → no reliable sub-pixel shift
        return 0
    offset = 0.5 * (left - right) / denominator
    return offset

def snap_intersections(intersections,
                       img,
                       horizontal_data,
                       vertical_data,
                       search_radius=25,
                       confidence_threshold=0.01,
                       bias_factor=0.6):
    """
    Local version – detects corners only in a small patch around each
    calculated intersection instead of on the whole image.
    Keeps the same row-wise (bottom → top), left-to-right order,
    exclusion rule and directional bias.
    Bias is now generalized to follow the orientation of the intersecting lines
    instead of fixed left/down directions.
    """
    # Convert to grayscale once
    if len(img.shape) == 3:
        gray = np.dot(img, [0.299, 0.587, 0.114])
    else:
        gray = img.astype(float)

    # Group intersections by horizontal line number (the integer label)
    groups = defaultdict(list)
    for inter in intersections:
        v_label, h_label_str = inter['name'].split('-')
        h_label = int(h_label_str)
        groups[h_label].append(inter)

    # Average y per row – used to process rows from bottom to top
    group_ys = {
        h: np.mean([inter['point'][1] for inter in group])
        for h, group in groups.items()
    }
    sorted_rows = sorted(group_ys.keys(), key=lambda h: group_ys[h], reverse=True)

    snapped = []
    used_corners = set()          # (x_int, y_int) tuples – already claimed

    for h_label in sorted_rows:
        row = groups[h_label]
        # left-to-right
        row.sort(key=lambda inter: inter['point'][0])

        for inter in row:
            px, py = inter['point']
            name = inter['name']

            # Parse labels to find the intersecting lines
            v_label, h_label_str = name.split('-')
            h_entry = next(entry for entry in horizontal_data if entry['label'] == int(h_label_str))
            v_entry = next(entry for entry in vertical_data if entry['label'] == v_label)
            h_p1, h_p2 = h_entry['points']
            v_p1, v_p2 = v_entry['points']

            # Compute preferred directions based on line orientations
            # For vertical: preferred "down" (positive y component after normalization)
            vec_v = np.array(v_p2) - np.array(v_p1)
            norm_v = np.linalg.norm(vec_v)
            if norm_v < 1e-6:
                preferred_v = np.array([0, 1.0])  # Fallback: positive y
            else:
                preferred_v = vec_v / norm_v
                if preferred_v[1] < 0:
                    preferred_v = -preferred_v  # Flip to make y positive

            # For horizontal: preferred "left" (negative x component after normalization)
            vec_h = np.array(h_p2) - np.array(h_p1)
            norm_h = np.linalg.norm(vec_h)
            if norm_h < 1e-6:
                preferred_h = np.array([-1.0, 0])  # Fallback: negative x
            else:
                preferred_h = vec_h / norm_h
                if preferred_h[0] > 0:
                    preferred_h = -preferred_h  # Flip to make x negative

            # Define search patch
            x0 = max(0, int(px - search_radius))
            x1 = min(gray.shape[1], int(px + search_radius + 1))
            y0 = max(0, int(py - search_radius))
            y1 = min(gray.shape[0], int(py + search_radius + 1))

            patch = gray[y0:y1, x0:x1]

            # Run Harris only on this tiny patch
            Ix = sobel(patch, axis=1)
            Iy = sobel(patch, axis=0)
            Ix2 = gaussian_filter(Ix**2, sigma=1.0)
            Iy2 = gaussian_filter(Iy**2, sigma=1.0)
            Ixy = gaussian_filter(Ix * Iy, sigma=1.0)

            det = Ix2 * Iy2 - Ixy**2
            trace = Ix2 + Iy2
            response = det - 0.05 * trace**2

            # Keep only strong responses
            if response.max() <= 0:
                snapped.append({'name': name, 'point': (px, py)})
                continue

            thresh = confidence_threshold * response.max()
            candidates = response > thresh

            # Local non-maximum suppression
            max_filt = maximum_filter(response, size=7)
            peaks = (response == max_filt) & candidates
            yy, xx = np.nonzero(peaks)

            best_dist = float('inf')
            best_pos = None

            for loc_y, loc_x in zip(yy, xx):
                # Sub-pixel refinement
                dx = 0.0
                dy = 0.0
                if 0 < loc_x < response.shape[1] - 1:
                    row_vals = response[loc_y, loc_x-1:loc_x+2]
                    dx = subpixel_peak(loc_x, row_vals)
                if 0 < loc_y < response.shape[0] - 1:
                    col_vals = response[loc_y-1:loc_y+2, loc_x]
                    dy = subpixel_peak(loc_y, col_vals)

                cx = x0 + loc_x + dx
                cy = y0 + loc_y + dy

                # Check if already used (exclusion)
                corner_key = (round(cx), round(cy))  # Use round for sub-pixel
                if corner_key in used_corners:
                    continue

                vec = np.array([cx - px, cy - py])

                raw_dist = np.linalg.norm(vec)

                if raw_dist >= search_radius:
                    continue

                # Apply line-oriented bias: penalise going against preferred directions
                biased_dist = raw_dist

                proj_v = np.dot(vec, preferred_v)
                if proj_v < 0:  # Against vertical preferred (e.g., "up" relative to line)
                    biased_dist += (-proj_v) * bias_factor

                proj_h = np.dot(vec, preferred_h)
                if proj_h < 0:  # Against horizontal preferred (e.g., "right" relative to line)
                    biased_dist += (-proj_h) * bias_factor

                if biased_dist < best_dist:
                    best_dist = biased_dist
                    best_pos = (cx, cy)
                    best_key = corner_key

            if best_pos is not None:
                used_corners.add(best_key)
                snapped.append({'name': name, 'point': best_pos})
            else:
                # No good corner found → keep original (calculated) point
                snapped.append({'name': name, 'point': (px, py)})

    # Return in the original alpha-numeric order
    snapped.sort(key=lambda item: (ord(item['name'].split('-')[0][0]),
                                   int(item['name'].split('-')[1])))
    return snapped

# Function to handle button press events (for panning)
def on_press(event):
    global panning, start_x, start_y, orig_xlim, orig_ylim
    if event.button == 3:  # Right mouse button for starting pan
        if event.inaxes == ax:
            panning = True
            start_x = event.xdata
            start_y = event.ydata
            orig_xlim = ax.get_xlim()
            orig_ylim = ax.get_ylim()

# Function to handle mouse motion events
def on_move(event):
    global panning, start_x, start_y, orig_xlim, orig_ylim
    if not panning:
        return
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    dx = event.xdata - start_x
    dy = event.ydata - start_y
    ax.set_xlim(orig_xlim[0] - dx, orig_xlim[1] - dx)
    ax.set_ylim(orig_ylim[0] - dy, orig_ylim[1] - dy)
    fig.canvas.draw()

# Function to handle button release events
def on_release(event):
    global panning
    if event.button == 3:
        panning = False

# Function to handle scroll events for zooming centered on cursor
def on_scroll(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    # Get current limits
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    # Determine scale factor: positive step for zoom in, negative for zoom out
    scale_factor = 1.1 ** -event.step  # Adjust 1.1 for zoom sensitivity
    # Calculate new dimensions
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    # Relative position of cursor within the axes
    relx = (cur_xlim[1] - x) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - y) / (cur_ylim[1] - cur_ylim[0])
    # Set new limits centered on cursor
    ax.set_xlim([x - new_width * (1 - relx), x + new_width * relx])
    ax.set_ylim([y - new_height * (1 - rely), y + new_height * rely])
    # Update the canvas
    fig.canvas.draw()

# Connect the events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)

# Calculate intersections
print("Calculating intersections from lines...")
intersections = calculate_intersections(horizontal_data, vertical_data)
print("Snapping intersections to real checkerboard corners...")
snapped_intersections = snap_intersections(
    intersections,
    img,
    horizontal_data,
    vertical_data,
    search_radius=25,           # Adjust if needed – start with 20–30
    confidence_threshold=0.1,  # Lower = more corners considered
    bias_factor=0.6             # 0.5–0.8 works well; higher = stronger downward/left bias
)

# Plot the horizontal lines and labels
for entry in horizontal_data:
    (x1, y1), (x2, y2) = entry['points']
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.text(mid_x, mid_y, str(entry['label']), color='blue', fontsize=12)

# Plot the vertical lines and labels
for entry in vertical_data:
    (x1, y1), (x2, y2) = entry['points']
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    ax.text(mid_x, mid_y, entry['label'], color='blue', fontsize=12)

# Plot the intersections as green dots
for inter in intersections:
    x, y = inter['point']
    # ax.plot(x, y, 'go', markersize=5, alpha=0.5)
    # ax.text(x+5, y+5, inter['name'], color='yellow', fontsize=8, alpha=0.4)

# Plot the snapped (final) points in bright cyan for visibility
for inter in snapped_intersections:
    x, y = inter['point']
    ax.plot(x, y, 'co', markersize=5)
    ax.text(x+5, y+5, inter['name'], color='yellow', fontsize=8)




# Display the plot
plt.show()

# Save intersections to JSON after closing the window
base_name = Path(json_path).stem.replace('_lines', '')
output_file = f"{base_name}_intersections.json"
data_to_save = {
    "image": image_path,
    "intersections": [
        {"name": inter['name'], "point": list(inter['point'])}
        for inter in intersections
    ]
}
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, indent=2)
print(f"Intersections saved to: {output_file}")

# Save the snapped intersections
base_name = Path(json_path).stem.replace('_lines', '')
output_file = f"{base_name}_intersections_snapped.json"
data_to_save = {
    "image": image_path,
    "source_lines_file": json_path,
    "intersections": [
        {"name": inter['name'], "point": list(inter['point'])}
        for inter in snapped_intersections
    ]
}
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, indent=2)

print(f"Snapped intersections saved to: {output_file}")

# Report
print(f"   Total points: {len(snapped_intersections)}")
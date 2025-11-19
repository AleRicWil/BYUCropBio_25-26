import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from pathlib import Path

# Load the image (replace 'your_image.jpg' with the actual file path)
image_path = 'right.jpg'
img = mpimg.imread(image_path)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title('Interactive Image Viewer')

# Lists and data structures
click_points = []
horizontal_lines = []
vertical_lines = []
mode = 'horizontal'  # Start with horizontal
direction_switched = False
original_step = None
current_step = None
current_base_label = None
len_at_switch = None
point_artists = []

# Variables for panning
panning = False
start_x = None
start_y = None
orig_xlim = None
orig_ylim = None

# Function to handle button press events
def on_press(event):
    global panning, start_x, start_y, orig_xlim, orig_ylim, original_step, current_step, current_base_label, direction_switched
    if event.button == 1:  # Left mouse button for adding points
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            x, y = snap_to_edge(img, x, y, mode)
            # Add blue dot
            dot = ax.plot(x, y, 'bo', markersize=5)[0]
            point_artists.append(dot)
            # Record the point
            click_points.append((x, y))
            # Every two clicks, connect with a thin red line
            if len(click_points) % 2 == 0:
                (x1, y1), (x2, y2) = click_points[-2:]
                line_artist = ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)[0]
                line = ((x1, y1), (x2, y2))
                if mode == 'horizontal':
                    len_before = len(horizontal_lines)
                    if len_before == 0:
                        print('The integer label is the number below the line')
                        label_str = input("Enter integer label for bottom horizontal line: ")
                        label = int(label_str)
                        print('Select the second line, the one just above the bottom line')
                    elif len_before == 1:
                        label_str = input("Enter integer label for second horizontal line: ")
                        label = int(label_str)
                        print('Continue to top of image. Then press "v" to do vertical lines. Labels are infered.')
                    else:
                        step = horizontal_lines[1]['label'] - horizontal_lines[0]['label']
                        label = horizontal_lines[-1]['label'] + step
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    text_artist = ax.text(mid_x, mid_y, str(label), color='red', fontsize=12)
                    horizontal_lines.append({'line': line, 'label': label, 'artists': [point_artists[-2], point_artists[-1], line_artist, text_artist]})
                elif mode == 'vertical':
                    len_before = len(vertical_lines)
                    if len_before == 0:
                        print('The character label is the char to the right of the line')
                        label_str = input("Enter character label for middle vertical line: ")
                        label = label_str[0] if label_str else ''
                        current_base_label = label
                        print('Select the next vertical line just to the right')
                    elif len_before == 1:
                        label_str = input("Enter character label (case sensitive) for second vertical line (start of first half): ")
                        label = label_str[0] if label_str else ''
                        original_step = ord(label) - ord(vertical_lines[0]['label'])
                        current_step = original_step
                        current_base_label = label
                        print('Select vertical lines in same direction to end of image. Then press "d" to do other direction')
                    else:
                        label = chr(ord(current_base_label) + current_step)
                        current_base_label = label
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    text_artist = ax.text(mid_x, mid_y, label, color='yellow', fontsize=12)
                    vertical_lines.append({'line': line, 'label': label, 'artists': [point_artists[-2], point_artists[-1], line_artist, text_artist]})
                # Update the canvas
                fig.canvas.draw()
    elif event.button == 3:  # Right mouse button for starting pan
        if event.inaxes == ax:
            panning = True
            start_x = event.xdata
            start_y = event.ydata
            orig_xlim = ax.get_xlim()
            orig_ylim = ax.get_ylim()

def snap_to_edge(img, x, y, mode, search_win=10, avg_win=5, sigma=1.0):
    """
    Snaps the click point to the nearest edge in the specified mode.
    
    Parameters:
    - img: numpy array of the image (h, w, 3) with values in [0, 1]
    - x, y: float coordinates of the click
    - mode: 'horizontal' or 'vertical'
    - search_win: int, search window size around the click
    - avg_win: int, averaging window size perpendicular to the snap direction
    - sigma: float, Gaussian sigma for smoothing the signal
    
    Returns:
    - snapped_x, snapped_y: float coordinates snapped to the edge
    """
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        return x, y  # Out of bounds, return original
    
    # Cache the grayscale image using function attribute (assumes same img each call)
    if not hasattr(snap_to_edge, 'gray') or id(snap_to_edge.img) != id(img):
        snap_to_edge.gray = np.dot(img, [0.299, 0.587, 0.114])
        snap_to_edge.img = img  # Store img reference for id check
    
    gray = snap_to_edge.gray
    ix, iy = int(round(x)), int(round(y))
    
    if mode == 'horizontal':
        # Snap y to nearest horizontal edge, keep x
        left = max(0, ix - avg_win)
        right = min(gray.shape[1], ix + avg_win + 1)
        slice_gray = gray[:, left:right]
        signal = np.mean(slice_gray, axis=1)
        signal = gaussian_filter1d(signal, sigma)
        
        start = max(0, iy - search_win)
        end = min(gray.shape[0], iy + search_win + 1)
        local_signal = signal[start:end]
        
        if len(local_signal) < 2:
            return x, y
        
        dy = np.diff(local_signal)
        abs_dy = np.abs(dy)
        peak_idx = np.argmax(abs_dy)
        
        snapped_y = start + peak_idx
        if 0 < peak_idx < len(abs_dy) - 1:
            a = abs_dy[peak_idx - 1]
            b = abs_dy[peak_idx]
            c = abs_dy[peak_idx + 1]
            denominator = a - 2 * b + c
            if denominator != 0:
                sub = 0.5 * (a - c) / denominator
                snapped_y += sub
        
        return x, snapped_y
    
    elif mode == 'vertical':
        # Snap x to nearest vertical edge, keep y
        top = max(0, iy - avg_win)
        bottom = min(gray.shape[0], iy + avg_win + 1)
        slice_gray = gray[top:bottom, :]
        signal = np.mean(slice_gray, axis=0)
        signal = gaussian_filter1d(signal, sigma)
        
        start = max(0, ix - search_win)
        end = min(gray.shape[1], ix + search_win + 1)
        local_signal = signal[start:end]
        
        if len(local_signal) < 2:
            return x, y
        
        dx = np.diff(local_signal)
        abs_dx = np.abs(dx)
        peak_idx = np.argmax(abs_dx)
        
        snapped_x = start + peak_idx
        if 0 < peak_idx < len(abs_dx) - 1:
            a = abs_dx[peak_idx - 1]
            b = abs_dx[peak_idx]
            c = abs_dx[peak_idx + 1]
            denominator = a - 2 * b + c
            if denominator != 0:
                sub = 0.5 * (a - c) / denominator
                snapped_x += sub
        
        return snapped_x, y
    
    return x, y

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

# Function for key press to switch mode
def on_key(event):
    global mode, direction_switched, current_step, original_step, current_base_label
    if event.key == 'h':
        mode = 'horizontal'
    elif event.key == 'v':
        mode = 'vertical'
        print('Select vertical line near the middle of the image. Note its character label before clicking.')
    elif event.key == 'd' and mode == 'vertical':
        if not direction_switched and len(vertical_lines) >= 2:
            len_at_switch = len(vertical_lines)
            direction_switched = True
            current_step = -original_step
            current_base_label = vertical_lines[0]['label']
            print("Switched to the other direction. The next vertical line will be labeled starting from the other side of the middle.")
            print('Select to end of image and press "q"')
    elif event.key == 'u':
        undo_last_line()

def undo_last_line():
    global current_base_label, direction_switched, current_step, original_step, len_at_switch
    if mode == 'horizontal':
        if horizontal_lines:
            last_entry = horizontal_lines.pop()
            for artist in last_entry.get('artists', []):
                artist.remove()
            if click_points:
                click_points.pop()
                point_artists.pop()
            if click_points:
                click_points.pop()
                point_artists.pop()
    elif mode == 'vertical':
        if vertical_lines:
            last_entry = vertical_lines.pop()
            for artist in last_entry.get('artists', []):
                artist.remove()
            if click_points:
                click_points.pop()
                point_artists.pop()
            if click_points:
                click_points.pop()
                point_artists.pop()
            if vertical_lines:
                current_base_label = vertical_lines[-1]['label']
            else:
                current_base_label = None
            if direction_switched and len_at_switch is not None and len(vertical_lines) < len_at_switch:
                direction_switched = False
                current_step = original_step
    fig.canvas.draw()

# Connect the events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

# Display the plot (the toolbar provides additional controls)
print('=' * 60)
print('KEYBOARD SHORTCUTS')
print('  h → horizontal mode   |   v → vertical mode')
print('  d → (vertical only) switch direction after one half')
print('  u → UNDO last line (removes line, points, and label)')
print('  Right-click drag → pan   |   Scroll wheel → zoom at cursor')
print('=' * 60)
print('Select left/right points of bottom line. Get close, then progam snaps to edge')
print('Always select as close to image border as possible so lines intersect at more corners')
plt.show()

# Build filename:  left.jpg  →  left_lines.json
base_name = Path(image_path).stem
output_file = f"{base_name}_lines.json"

data_to_save = {
    "image": image_path,
    "horizontal_lines": [
        {"label": entry["label"],
         "points": entry["line"]}   # ((x1,y1), (x2,y2))
        for entry in horizontal_lines
    ],
    "vertical_lines": [
        {"label": entry["label"],
         "points": entry["line"]}
        for entry in vertical_lines
    ]
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, indent=2)

print(f"Data saved to: {output_file}")
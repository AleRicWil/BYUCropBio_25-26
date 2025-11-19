import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Load the line JSON data (replace with your actual JSON file path)
json_path = 'right_lines.json'
with open(json_path, 'r') as f:
    data = json.load(f)

image_path = data['image']
horizontal_data = data['horizontal_lines']
vertical_data = data['vertical_lines']

# Load the snapped JSON data (replace with your actual JSON file path)
json_path = 'right_intersections_snapped.json'
with open(json_path, 'r') as f:
    data = json.load(f)

image_path = data['image']
intersections = data['intersections']  # List of {'name': , 'point': [x, y]}

# Load the image
img = mpimg.imread(image_path)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title('Snapped Intersections Viewer - Click to select and move points')

# Variables for panning and selection
panning = False
deleting = False
start_x = None
start_y = None
orig_xlim = None
orig_ylim = None
selected_idx = None
point_artists = []
labels = []

# Function to handle button press events
def on_press(event):
    global panning, start_x, start_y, orig_xlim, orig_ylim, selected_idx
    if event.button == 1:  # Left mouse button for selection/moving
        if event.xdata is not None and event.ydata is not None:
            click_x, click_y = event.xdata, event.ydata
            if selected_idx is None:
                # Find closest point if within threshold
                min_dist = float('inf')
                closest_idx = None
                for i, inter in enumerate(intersections):
                    px, py = inter['point']
                    dist = np.hypot(px - click_x, py - click_y)
                    if dist < min_dist and dist < 20:  # Adjust threshold as needed
                        min_dist = dist
                        closest_idx = i
                if closest_idx is not None:
                    selected_idx = closest_idx
                    name = intersections[selected_idx]['name']
                    print(f"Selected point: {name} at {intersections[selected_idx]['point']}")
                else:
                    print("No point selected near click position.")
            else:
                # Move the selected point to new position
                name = intersections[selected_idx]['name']
                old_point = intersections[selected_idx]['point']
                intersections[selected_idx]['point'] = [click_x, click_y]
                point_artists[selected_idx].set_data([click_x], [click_y])
                labels[selected_idx].set_position((click_x, click_y))
                print(f"Moved point {name} from {old_point} to {[click_x, click_y]}")
                selected_idx = None  # Deselect after move
                fig.canvas.draw()
    elif event.button == 3:  # Right mouse button for starting pan
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

def on_key(event):
    global selected_idx, deleting
    if event.key == 'd' and selected_idx is not None:
        name = intersections[selected_idx]['name']
        point_artists[selected_idx].remove()
        labels[selected_idx].remove()
        del intersections[selected_idx]
        del point_artists[selected_idx]
        del labels[selected_idx]
        print(f"Deleted point: {name}")
        selected_idx = None
        fig.canvas.draw()

# Connect the events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)

# Plot the horizontal lines and labels
for entry in horizontal_data:
    (x1, y1), (x2, y2) = entry['points']
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

# Plot the vertical lines and labels
for entry in vertical_data:
    (x1, y1), (x2, y2) = entry['points']
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

# Plot the snapped points and labels
for inter in intersections:
    x, y = inter['point']
    artist = ax.plot(x, y, 'co', markersize=6)[0]
    point_artists.append(artist)
    label = ax.text(x, y, inter['name'], color='white', fontsize=8, ha='center', va='bottom')
    labels.append(label)

# Display the plot
print("Instructions: Left-click near a point to select it (confirmed in console). Then left-click the new position to move it. Changes are saved on close.")
plt.show()

# Save corrected intersections after closing the window
base_name = Path(json_path).stem.replace('_snapped', '')
output_file = f"{base_name}_corrected.json"
data_to_save = {
    "image": image_path,
    "intersections": [
        {"name": inter['name'], "point": inter['point']}
        for inter in intersections
    ]
}
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, indent=2)
print(f"Corrected intersections saved to: {output_file}")
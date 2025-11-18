import pandas as pd
import numpy as np
import os
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from matplotlib.patches import Rectangle
import argparse  # NEW: For CLI args

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

# NEW: Parse command-line arguments
# Key Concept: Configurable scripts via CLI for flexibility
parser = argparse.ArgumentParser(description="Saffron Field Viewer: Plot raw or filtered/grouped GPS data.")
parser.add_argument('--mode', type=str, choices=['raw', 'filtered'], default='filtered',
                    help="Plot mode: 'raw' for unfiltered points, 'filtered' for processed cells (default: filtered)")
args = parser.parse_args()

# -------------------------------
# Load data
# -------------------------------
csv_path = r'C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\captures10-27_11pm.csv'
df = pd.read_csv(csv_path, low_memory=False)

df = df.dropna(subset=['latitude', 'longitude'])
df = df[(df['latitude'] != '') & (df['longitude'] != '')]

def get_image_path(row):
    camera = row['camera']
    timestamp = row['timestamp']
    filename = f"{camera}_color_{timestamp}.jpg"
    folder = rf"C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\{camera}"
    return os.path.join(folder, filename)

df['image_path'] = df.apply(get_image_path, axis=1)

# Compute raw grouped data (for raw mode)
raw_grouped = df.groupby(['latitude', 'longitude'])['image_path'].agg(list).reset_index()

# Key Concept: Convert meters to degrees for binning (approximate geodesic distances)
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
lat_avg = df['latitude'].mean()  # Use average lat for longitude scaling
meters_per_deg_lat = 111319.9  # WGS84 equatorial, approximate
delta_lat = 0.5 / meters_per_deg_lat  # 0.5m in lat
meters_per_deg_lon = 111319.9 * math.cos(math.radians(lat_avg))
half_delta_lon = 0.5 / meters_per_deg_lon  # 0.5m for half-width

# Identify initial fine clusters via rounding (handles GPS noise ~0.1m)
df['lon_rounded'] = df['longitude'].round(6)  # Precision to ~0.1m
initial_centers = df.groupby('lon_rounded')['longitude'].mean().sort_values().values

# Regroup fine clusters to merge splits within true columns
merge_threshold_m = 0.2  # Merge if <0.2m apart (tolerant to GPS noise, adjustable)
merge_threshold_deg = merge_threshold_m / meters_per_deg_lon
merged_groups = []
current_group = [initial_centers[0]]
for lon in initial_centers[1:]:
    if lon - current_group[-1] < merge_threshold_deg:
        current_group.append(lon)
    else:
        merged_groups.append(current_group)
        current_group = [lon]
merged_groups.append(current_group)

# Compute true column centers as averages of merged groups
cluster_centers = [np.mean(group) for group in merged_groups]
cluster_centers = sorted(cluster_centers)
left_most_center = cluster_centers[0]
right_most_center = cluster_centers[-1]

# Compute diffs for spacing
cluster_diffs = np.diff(cluster_centers)
print(np.round(cluster_diffs * meters_per_deg_lon, 2))  # Inspect true spacings in meters

# NEW: Filter anomalies (GPS drift/jumps) only if mode == 'filtered'
if args.mode == 'filtered':
    # Outlier rejection - discard points far from any true column center
    if len(cluster_centers) > 1:
        avg_spacing = np.mean(cluster_diffs)
        threshold_lon = avg_spacing * 0.25  # Use 0.25x average spacing (wide tolerance for 'straight' motion)
    else:
        threshold_lon = half_delta_lon * 2  # Fallback if single column

    def nearest_center_dist(lon, centers):
        return min(abs(lon - c) for c in centers)

    df['nearest_dist'] = df['longitude'].apply(lambda x: nearest_center_dist(x, cluster_centers))
    df = df[df['nearest_dist'] < threshold_lon].drop(columns=['nearest_dist'])

# Longitude bin edges: Center 1m wide bins on each cluster center, gaps fill the rest (for filtered)
bin_edges_lon = [lon_min]
for i, center in enumerate(cluster_centers):
    bin_edges_lon.append(center - half_delta_lon)
    bin_edges_lon.append(center + half_delta_lon)
bin_edges_lon.append(lon_max)
bin_edges_lon = sorted(set(bin_edges_lon))  # Dedup and sort

# Latitude bin edges: Standard uniform coverage
bin_edges_lat = np.arange(lat_min, lat_max + delta_lat, delta_lat)

if args.mode == 'filtered':
    # Assign bins
    df['lat_bin'] = pd.cut(df['latitude'], bins=bin_edges_lat, labels=range(len(bin_edges_lat)-1))
    df['lon_bin'] = pd.cut(df['longitude'], bins=bin_edges_lon, labels=range(len(bin_edges_lon)-1))

    # Group images by bins for popups
    grouped = df.groupby(['lat_bin', 'lon_bin'], observed=True)['image_path'].agg(list).reset_index()

    # Compute bin centers (in degrees)
    def get_bin_center(bin_id, edges):
        if pd.isna(bin_id):
            return np.nan
        bin_id = int(bin_id)
        return (edges[bin_id] + edges[bin_id + 1]) / 2

    grouped['center_lat'] = grouped['lat_bin'].apply(get_bin_center, edges=bin_edges_lat)
    grouped['center_lon'] = grouped['lon_bin'].apply(get_bin_center, edges=bin_edges_lon)

    # Drop any NaN centers or empty paths (edge cases)
    grouped = grouped.dropna(subset=['center_lat', 'center_lon'])
    grouped = grouped.dropna(subset=['image_path'])

    # Normalized bin edges for overlay
    norm_bin_edges_lat = (bin_edges_lat - lat_min) / (lat_max - lat_min + 1e-12)
    norm_bin_edges_lon = (bin_edges_lon - lon_min) / (lon_max - lon_min + 1e-12)

# Prepare GPS points for plotting (uses df, which is filtered if mode=='filtered')
gps_grouped = df.groupby(['latitude', 'longitude']).agg({'image_path': 'count'}).reset_index().rename(columns={'image_path': 'count'})
gps_lats = gps_grouped['latitude'].values
gps_lons = gps_grouped['longitude'].values
norm_gps_lats = (gps_lats - lat_min) / (lat_max - lat_min + 1e-12)
norm_gps_lons = (gps_lons - lon_min) / (lon_max - lon_min + 1e-12)

# ----------------------------------------------------------------
# GUI class (now accepts mode)
# ----------------------------------------------------------------
class FieldViewer(tk.Tk):
    def __init__(self, mode='filtered'):
        super().__init__()
        self.mode = mode
        self.title("Saffron Field Viewer – Desktop (No Browser)")
        self.geometry("1000x700")

        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Click any point/cell to view photos", fontsize=14)
        self.ax.set_xlabel("Normalized Longitude")
        self.ax.set_ylabel("Normalized Latitude")

        # Plot non-clickable GPS dots, colored by image count (common to both modes)
        num_images = gps_grouped['count'].values
        self.scatter = self.ax.scatter(norm_gps_lons, norm_gps_lats,
                                       s=10,                    # Fixed, readable size
                                       c=num_images,            # Color = number of images
                                       cmap='viridis',          # Yellow = many, purple/blue = few
                                       alpha=0.8,
                                       edgecolors='white',
                                       linewidth=0.5,
                                       picker=(self.mode == 'raw'))  # Clickable only in raw mode

        if self.mode == 'filtered':
            # Overlay grid lines for visual reference
            for edge in norm_bin_edges_lat:
                self.ax.axhline(edge, color='gray', ls='--', alpha=0.3)
            for edge in norm_bin_edges_lon:
                self.ax.axvline(edge, color='gray', ls='--', alpha=0.3)

            # Add clickable invisible rectangles for each grid cell
            self.rects = []
            for _, row in grouped.iterrows():
                lat_bin_id = int(row['lat_bin'])
                lon_bin_id = int(row['lon_bin'])
                ll_y = norm_bin_edges_lat[lat_bin_id]
                ll_x = norm_bin_edges_lon[lon_bin_id]
                height = norm_bin_edges_lat[lat_bin_id + 1] - ll_y
                width = norm_bin_edges_lon[lon_bin_id + 1] - ll_x
                rect = Rectangle((ll_x, ll_y), width, height,
                                 facecolor='none',  # Invisible fill
                                 edgecolor='none',  # No edges (grid lines are separate)
                                 alpha=0,           # Fully transparent
                                 picker=True)
                rect.images = row['image_path']
                rect.center_lat = row['center_lat']
                rect.center_lon = row['center_lon']
                self.ax.add_patch(rect)
                self.rects.append(rect)  # Keep reference if needed

        # Annotate left-most and right-most column averages (common)
        norm_left = (left_most_center - lon_min) / (lon_max - lon_min + 1e-12)
        norm_right = (right_most_center - lon_min) / (lon_max - lon_min + 1e-12)
        self.ax.axvline(norm_left, color='red', ls='--', alpha=0.5)
        self.ax.axvline(norm_right, color='red', ls='--', alpha=0.5)
        self.ax.annotate(f'{left_most_center:.7f}',
                         xy=(norm_left, 1), xycoords='axes fraction',
                         xytext=(-5, 5), textcoords='offset points',
                         ha='right', va='bottom', rotation=90, fontsize=9)
        self.ax.annotate(f'{right_most_center:.7f}',
                         xy=(norm_right, 1), xycoords='axes fraction',
                         xytext=(5, 5), textcoords='offset points',
                         ha='left', va='bottom', rotation=90, fontsize=9)

        # Embed into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar (zoom, pan, etc.)
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Click handler
        self.canvas.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event):
        artist = event.artist
        if self.mode == 'raw' and artist == self.scatter:
            ind = event.ind[0]
            image_paths = raw_grouped['image_path'].iloc[ind]
            real_lat = raw_grouped['latitude'].iloc[ind]
            real_lon = raw_grouped['longitude'].iloc[ind]
            self.show_images(image_paths, f"Lat {real_lat:.7f} | Lon {real_lon:.7f}")
        elif self.mode == 'filtered' and isinstance(artist, Rectangle):
            image_paths = artist.images
            real_lat = artist.center_lat
            real_lon = artist.center_lon
            self.show_images(image_paths, f"Cell Center Lat {real_lat:.7f} | Lon {real_lon:.7f}")

    def show_images(self, paths, title):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("500x700")

        # Scrollable canvas
        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Multi-level sorting & grouping
        camera_order = {"RS1": 0, "RS2": 1, "RS3": 2}

        def sort_key(path):
            basename = os.path.basename(path)
            parts = basename.split("_color_")
            if len(parts) == 2:
                camera = parts[0]
                ts = parts[1].rstrip(".jpg")
                return (camera_order.get(camera, 99), ts)
            return (99, basename)

        sorted_paths = sorted(paths, key=sort_key)

        prev_camera = None
        for path in sorted_paths:
            basename = os.path.basename(path)
            parts = basename.split('_color_')
            if len(parts) == 2:
                camera = parts[0]
                timestamp_str = parts[1].replace('.jpg', '')
                date_part = timestamp_str[:8]
                time_part = timestamp_str[9:15]
                millis = timestamp_str[16:19]
                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}.{millis}"
                label_text = f"Camera: {camera} | Date: {formatted_date} | Time: {formatted_time} Z"
            else:
                camera = "Unknown"
                label_text = "Metadata not available"

            # Visual group header when camera changes
            if prev_camera != camera:
                ttk.Label(scrollable_frame,
                          text=f"===== {camera} =====",
                          font=("Arial", 12, "bold"),
                          foreground="darkblue").pack(pady=12)
                prev_camera = camera

            # Metadata label
            ttk.Label(scrollable_frame, text=label_text, font=("Arial", 10, "bold")).pack(pady=2)

            # Image
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img = img.resize((480, int(480 * img.height / img.width)), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    lbl = ttk.Label(scrollable_frame, image=photo)
                    lbl.image = photo
                    lbl.pack(pady=8)
                except Exception as e:
                    ttk.Label(scrollable_frame, text=f"Error: {e}").pack()
            else:
                ttk.Label(scrollable_frame, text=f"Missing: {basename}").pack()

# ----------------------------------------------------------------
# Run the app
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data and starting desktop viewer...")
    app = FieldViewer(mode=args.mode)
    app.mainloop()
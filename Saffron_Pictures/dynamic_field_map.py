import pandas as pd
import numpy as np
import os
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from matplotlib.patches import Rectangle
import argparse  # NEW: For CLI args
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import threading  # NEW: For background preloading

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
parser.add_argument('--lat_grid', type=float, default=1.0,
                    help="Latitude (NS) grid size in meters (min 0.1; if > half column height, use single cell per column)")
args = parser.parse_args()

# NEW: Global sort helpers (moved from show_images for reuse in pre-sorting)
camera_order = {"RS1": 0, "RS2": 1, "RS3": 2}

def sort_key(path):
    basename = os.path.basename(path)
    parts = basename.split("_color_")
    if len(parts) == 2:
        camera = parts[0]
        ts = parts[1].rstrip(".jpg")
        return (camera_order.get(camera, 99), ts)
    return (99, basename)

# -------------------------------
# Load data
# -------------------------------
csv_path = r'C:\Users\Alex R. Williams\Pictures\Saffron Data\10-27_11pm\captures10-27_11pm.csv'
# csv_path = r'C:\Users\Alex R. Williams\Pictures\Saffron Data\calibrate\captures.csv'
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
raw_grouped = df.groupby(['latitude', 'longitude'])['image_path'].agg(lambda x: sorted(list(x), key=sort_key)).reset_index()  # NEW: Pre-sort

# Key Concept: Convert meters to degrees for binning (approximate geodesic distances)
dEW = 0.5
dNS = max(args.lat_grid, 0.1)
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
lat_avg = df['latitude'].mean()  # Use average lat for longitude scaling
meters_per_deg_lat = 111319.9  # WGS84 equatorial, approximate
delta_lat = dNS / meters_per_deg_lat  # 0.5m in lat
meters_per_deg_lon = 111319.9 * math.cos(math.radians(lat_avg))
half_delta_lon = dEW / meters_per_deg_lon  # 0.5m for half-width

# -------------------------------
# ROBUST COLUMN CENTER DETECTION VIA HISTOGRAM PEAKS
# -------------------------------
# 5–10 cm bins give excellent separation of noise vs. real drift
bin_width_m = 0.07          # ~7 cm works extremely well in practice; you can make it 0.05–0.1
bin_width_deg = bin_width_m / meters_per_deg_lon

left = lon_min - 0.001
right = lon_max + 0.001
bins = np.arange(left, right, bin_width_deg)

hist, bin_edges = np.histogram(df['longitude'], bins=bins)

# Slight Gaussian smoothing merges GPS noise but keeps real peaks sharp
hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)  # sigma ≈ 14 cm smoothing

# Adaptive prominence – columns with ≥ ~8–10% of the tallest peak are kept
max_height = hist_smooth.max()
min_prominence = max(1, max_height * 0.08)   # tweak 0.08 only if needed

# minimum spacing ≈ 30 cm to kill spurious peaks
min_distance_bins = int(0.30 / bin_width_m)   # ~4–5 bins

peaks, _ = signal.find_peaks(hist_smooth, prominence=min_prominence, distance=min_distance_bins)

cluster_centers = bin_edges[peaks] + bin_width_deg / 2

cluster_centers = sorted(cluster_centers)
cluster_centers_meters = np.array(cluster_centers) * meters_per_deg_lon
cluster_centers_meters -= np.min(cluster_centers_meters)
cluster_diffs = np.diff(cluster_centers)
cluster_diffs_meters = np.diff(cluster_centers_meters)

print(f"Detected {len(cluster_centers)} columns using histogram peaks")
print("Centers W-E (m):", [f"{c:.2f}" for c in cluster_centers_meters])
print("Spacings W-E (m):", np.round(cluster_diffs_meters, 3))
print("Average Spacing (m):", f'{np.mean(cluster_diffs_meters):.2f}')
left_most_center = cluster_centers[0]
right_most_center = cluster_centers[-1]

# -------------------------------
# Filter anomalies only in filtered mode
# -------------------------------
if args.mode == 'filtered':
    # Outlier rejection - discard points far from any true column center
    cluster_diffs = np.diff(cluster_centers)          # <--- make sure this line exists (add if missing)
    if len(cluster_centers) > 1:
        avg_spacing_deg = np.mean(cluster_diffs)
        threshold_lon_deg = avg_spacing_deg * 0.25      # 25% of avg spacing = very tolerant
    else:
        threshold_lon_deg = 0.00001  # ~1 meter fallback, generous for single column

    df['nearest_dist'] = df['longitude'].apply(lambda lon: min(abs(lon - c) for c in cluster_centers))
    df = df[df['nearest_dist'] < threshold_lon_deg].drop(columns=['nearest_dist'])

    # NEW: Temporal drift rejection – cut off hooks/tails after large time gaps in each column
    if len(cluster_centers) > 0:
        centers_arr = np.array(cluster_centers)
        df['assigned_center'] = df['longitude'].apply(lambda lon: centers_arr[np.argmin(np.abs(centers_arr - lon))])

        keep_rows = []

        for _ , group in df.groupby('assigned_center'):
            if len(group) < 2:
                keep_rows.extend(group.index)
                continue

            # Reconstruct ISO datetime string from your filename convention
            group['dt'] = pd.to_datetime(
                group['timestamp'].apply(
                    lambda x: f"{x[0:4]}-{x[4:6]}-{x[6:8]} {x[9:11]}:{x[11:13]}:{x[13:15]}.{x[16:19]}"
                )
            )

            # Sort chronologically (preserves original df indices!)
            sorted_group = group.sort_values('dt')

            # Detect gaps >15 s and label segments
            is_large_gap = (sorted_group['dt'].diff() > pd.Timedelta(seconds=15)).fillna(False)
            sorted_group['segment_id'] = is_large_gap.cumsum()

            # Find the segment with the most points → this is almost always the real pass
            seg_sizes = sorted_group['segment_id'].value_counts()
            main_segment_id = seg_sizes.idxmax()

            # Keep only points from the largest segment
            keep_indices = sorted_group[sorted_group['segment_id'] == main_segment_id].index.tolist()
            keep_rows.extend(keep_indices)

        df = df.loc[keep_rows].drop(columns=['assigned_center', 'dt'], errors='ignore')

# Longitude bin edges: Center 1m wide bins on each cluster center, gaps fill the rest (for filtered)
bin_edges_lon = [lon_min]
for i, center in enumerate(cluster_centers):
    bin_edges_lon.append(center - half_delta_lon)
    bin_edges_lon.append(center + half_delta_lon)
bin_edges_lon.append(lon_max)
bin_edges_lon = sorted(set(bin_edges_lon))  # Dedup and sort

# Compute field NS height in meters for adaptive binning
height_m = (lat_max - lat_min) * meters_per_deg_lat
half_height_m = height_m / 2.0

if dNS > half_height_m and height_m > 0:
    # Force single cell per column (full height)
    bin_edges_lat = np.array([lat_min, lat_max + 1e-10])  # Tiny epsilon for pd.cut to work
else:
    bin_edges_lat = np.arange(lat_min, lat_max + delta_lat, delta_lat)

if args.mode == 'filtered':
    # Assign bins
    df['lat_bin'] = pd.cut(df['latitude'], bins=bin_edges_lat, labels=range(len(bin_edges_lat)-1))
    df['lon_bin'] = pd.cut(df['longitude'], bins=bin_edges_lon, labels=range(len(bin_edges_lon)-1))

    # Group images by bins for popups
    grouped = df.groupby(['lat_bin', 'lon_bin'], observed=True)['image_path'].agg(lambda x: sorted(list(x), key=sort_key)).reset_index()  # NEW: Pre-sort

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

        # NEW: Caches for fast image display
        self.image_cache = {}  # path -> PhotoImage (shared across cells)
        self.preload_cache = {}  # (lat_bin, lon_bin) -> 'loading' or 'done'

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

        # NEW: Hover handler for preloading
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

    # NEW: Preload on hover (current cell + neighbors)
    def on_hover(self, event):
        if self.mode != 'filtered' or event.xdata is None or event.ydata is None:
            return

        # Find current bin (np.searchsorted for efficiency)
        lon_bin = np.searchsorted(norm_bin_edges_lon, event.xdata) - 1
        lat_bin = np.searchsorted(norm_bin_edges_lat, event.ydata) - 1

        if lon_bin < 0 or lat_bin < 0 or lon_bin >= len(norm_bin_edges_lon) - 1 or lat_bin >= len(norm_bin_edges_lat) - 1:
            return

        # Preload current + neighbors (3x3 grid, clamped to bounds)
        lat_range = range(max(0, lat_bin - 1), min(len(norm_bin_edges_lat) - 1, lat_bin + 2))
        lon_range = range(max(0, lon_bin - 1), min(len(norm_bin_edges_lon) - 1, lon_bin + 2))

        for lbin in lat_range:
            for obin in lon_range:
                key = (lbin, obin)
                if key in self.preload_cache:
                    continue  # Already loading/done

                self.preload_cache[key] = 'loading'
                row = grouped[(grouped['lat_bin'] == lbin) & (grouped['lon_bin'] == obin)]
                if row.empty:
                    del self.preload_cache[key]
                    continue

                paths = row['image_path'].iloc[0]  # Already pre-sorted

                def load_paths():
                    for path in paths:
                        if path not in self.image_cache and os.path.exists(path):
                            try:
                                img = Image.open(path)
                                h = int(480 * img.height / img.width)
                                img = img.resize((480, h), Image.LANCZOS)
                                photo = ImageTk.PhotoImage(img)
                                self.image_cache[path] = photo
                            except:
                                pass  # Silent fail, as in original
                    self.preload_cache[key] = 'done'

                threading.Thread(target=load_paths, daemon=True).start()

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
        print(f'Loading {len(paths)} photos...')
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1440x700")  # Widened for three columns

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

        # Group paths by camera (assumes RS1=left/0, RS2=center/1, RS3=right/2)
        camera_paths = {'RS1': [], 'RS2': [], 'RS3': []}
        other_paths = []  # Fallback for unknown cameras

        for path in paths:  # paths already pre-sorted globally
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

            if camera in camera_paths:
                camera_paths[camera].append((path, label_text))
            else:
                other_paths.append((path, label_text))

        # Set up grid columns with equal weight
        for col in range(3):
            scrollable_frame.grid_columnconfigure(col, weight=1)

        # Display each camera in its column
        cameras = ['RS1', 'RS2', 'RS3']
        for col, cam in enumerate(cameras):
            # Header
            ttk.Label(scrollable_frame,
                        text=f"===== {cam} =====",
                        font=("Arial", 12, "bold"),
                        foreground="darkblue").grid(row=0, column=col, pady=12, sticky="n")

            row = 1  # Start below header
            for path, label_text in camera_paths[cam]:
                # Metadata label
                ttk.Label(scrollable_frame, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=col, pady=2, sticky="n")
                row += 1

                # Image (use cache if available)
                if os.path.exists(path):
                    try:
                        if path in self.image_cache:
                            photo = self.image_cache[path]
                        else:
                            img = Image.open(path)
                            img = img.resize((480, int(480 * img.height / img.width)), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            self.image_cache[path] = photo
                        lbl = ttk.Label(scrollable_frame, image=photo)
                        lbl.image = photo
                        lbl.grid(row=row, column=col, pady=8, sticky="n")
                    except Exception as e:
                        ttk.Label(scrollable_frame, text=f"Error: {e}").grid(row=row, column=col)
                    row += 1
                else:
                    ttk.Label(scrollable_frame, text=f"Missing: {basename}").grid(row=row, column=col)
                    row += 1

        # If any unknown/other paths, dump them in the center column (or handle as needed)
        if other_paths:
            col = 1  # Center
            row = max([len(camera_paths[cam]) * 2 + 1 for cam in cameras]) + 1  # Below the tallest column
            ttk.Label(scrollable_frame,
                        text="===== Unknown =====",
                        font=("Arial", 12, "bold"),
                        foreground="darkblue").grid(row=row, column=col, pady=12, sticky="n")
            row += 1
            for path, label_text in other_paths:
                ttk.Label(scrollable_frame, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=col, pady=2, sticky="n")
                row += 1
                # Similar image loading as above...
                if os.path.exists(path):
                    try:
                        if path in self.image_cache:
                            photo = self.image_cache[path]
                        else:
                            img = Image.open(path)
                            img = img.resize((480, int(480 * img.height / img.width)), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            self.image_cache[path] = photo
                        lbl = ttk.Label(scrollable_frame, image=photo)
                        lbl.image = photo
                        lbl.grid(row=row, column=col, pady=8, sticky="n")
                    except Exception as e:
                        ttk.Label(scrollable_frame, text=f"Error: {e}").grid(row=row, column=col)
                    row += 1
                else:
                    ttk.Label(scrollable_frame, text=f"Missing: {basename}").grid(row=row, column=col)
                    row += 1

# ----------------------------------------------------------------
# Run the app
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data and starting desktop viewer...")
    app = FieldViewer(mode=args.mode)
    app.mainloop()
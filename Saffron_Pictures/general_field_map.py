import pandas as pd
import numpy as np
import os
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.patches import Rectangle
import argparse  # For CLI args, if needed
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import threading  # For background preloading

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

# Parse command-line arguments (optional, can remove if not needed)
parser = argparse.ArgumentParser(description="General Image Map Viewer")
parser.add_argument('--mode', type=str, choices=['raw', 'filtered'], default='filtered',
                    help="Plot mode: 'raw' or 'filtered' (default: filtered)")
parser.add_argument('--lat_grid', type=float, default=1.0,
                    help="Latitude grid size in meters")
args = parser.parse_args()

class GeneralImageMap(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("General Image Map Viewer")
        self.geometry("1200x800")

        # UI for folder selection
        self.folder_path = tk.StringVar()
        self.radius = tk.DoubleVar(value=0.5)  # Default radius 5 meters

        frame = ttk.Frame(self)
        frame.pack(pady=10)

        ttk.Label(frame, text="Folder:").grid(row=0, column=0, padx=5)
        self.entry = ttk.Entry(frame, textvariable=self.folder_path, width=50)
        self.entry.grid(row=0, column=1, padx=5)

        browse_btn = ttk.Button(frame, text="Browse...", command=self.browse_folder)
        browse_btn.grid(row=0, column=2, padx=5)

        load_btn = ttk.Button(frame, text="Load", command=self.load_data)
        load_btn.grid(row=0, column=3, padx=5)

        ttk.Label(frame, text="Radius (m):").grid(row=1, column=0, padx=5)
        radius_entry = ttk.Entry(frame, textvariable=self.radius, width=10)
        radius_entry.grid(row=1, column=1, padx=5, sticky='w')

        # Placeholder for plot
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data variables
        self.df = None
        self.grouped = None
        self.raw_grouped = None
        self.norm_bin_edges_lat = None
        self.norm_bin_edges_lon = None
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.meters_per_deg_lat = None
        self.meters_per_deg_lon = None
        self.combined_folder = None
        self.image_cache = {}
        self.preload_cache = {}
        self.flower_scatter = None

        # Event bindings (will connect after load)
        self.cid_pick = None
        self.cid_hover = None
        self.cid_click = None

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def load_data(self):
        folder = self.folder_path.get()
        if not folder:
            tk.messagebox.showerror("Error", "Select a folder first.")
            return

        self.combined_folder = os.path.join(folder, 'Combined_Original')
        csv_path = os.path.join(folder, 'captures.csv')
        flowers_path = os.path.join(folder, 'flowers.csv')

        if not os.path.exists(self.combined_folder) or not os.path.exists(csv_path):
            tk.messagebox.showerror("Error", "Folder must contain Combined_Original and captures.csv")
            return

        # Clear previous plot
        self.ax.clear()
        if self.cid_pick:
            self.canvas.mpl_disconnect(self.cid_pick)
        if self.cid_hover:
            self.canvas.mpl_disconnect(self.cid_hover)
        if self.cid_click:
            self.canvas.mpl_disconnect(self.cid_click)
        self.image_cache.clear()
        self.preload_cache.clear()

        # Load captures.csv
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'] != '') & (df['longitude'] != '')]

        # Get available frame_ids from Combined_Original
        available_files = [f for f in os.listdir(self.combined_folder) if f.endswith('_original.jpg')]
        available_frame_ids = [f.replace('_original.jpg', '') for f in available_files]

        # Filter to RS1 rows with matching frame_ids
        df = df[(df['camera'] == 'RS1') & (df['timestamp'].isin(available_frame_ids))]
        df['frame_id'] = df['timestamp']

        if df.empty:
            tk.messagebox.showerror("Error", "No matching data found.")
            return

        self.df = df

        # Compute bounds and conversions
        self.lat_min, self.lat_max = df['latitude'].min(), df['latitude'].max()
        self.lon_min, self.lon_max = df['longitude'].min(), df['longitude'].max()
        lat_avg = df['latitude'].mean()
        self.meters_per_deg_lat = 111319.9
        self.meters_per_deg_lon = 111319.9 * math.cos(math.radians(lat_avg))

        # Column detection (as before)
        bin_width_m = 0.07
        bin_width_deg = bin_width_m / self.meters_per_deg_lon
        left = self.lon_min - 0.001
        right = self.lon_max + 0.001
        bins = np.arange(left, right, bin_width_deg)
        hist, bin_edges = np.histogram(df['longitude'], bins=bins)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
        max_height = hist_smooth.max()
        min_prominence = max(1, max_height * 0.08)
        min_distance_bins = int(0.30 / bin_width_m)
        peaks, _ = signal.find_peaks(hist_smooth, prominence=min_prominence, distance=min_distance_bins)
        cluster_centers = bin_edges[peaks] + bin_width_deg / 2
        cluster_centers = sorted(cluster_centers)
        left_most_center = cluster_centers[0] if cluster_centers else self.lon_min
        right_most_center = cluster_centers[-1] if cluster_centers else self.lon_max

        # Filter anomalies if filtered
        if args.mode == 'filtered':
            if len(cluster_centers) > 1:
                cluster_diffs = np.diff(cluster_centers)
                avg_spacing_deg = np.mean(cluster_diffs)
                threshold_lon_deg = avg_spacing_deg * 0.25
            else:
                threshold_lon_deg = 0.00001

            df['nearest_dist'] = df['longitude'].apply(lambda lon: min(abs(lon - c) for c in cluster_centers))
            df = df[df['nearest_dist'] < threshold_lon_deg].drop(columns=['nearest_dist'])

            if len(cluster_centers) > 0:
                centers_arr = np.array(cluster_centers)
                df['assigned_center'] = df['longitude'].apply(lambda lon: centers_arr[np.argmin(np.abs(centers_arr - lon))])
                keep_rows = []
                for _, group in df.groupby('assigned_center'):
                    if len(group) < 2:
                        keep_rows.extend(group.index)
                        continue
                    group['dt'] = pd.to_datetime(group['timestamp'].apply(lambda x: f"{x[0:4]}-{x[4:6]}-{x[6:8]} {x[9:11]}:{x[11:13]}:{x[13:15]}.{x[16:19]}"))
                    sorted_group = group.sort_values('dt')
                    is_large_gap = (sorted_group['dt'].diff() > pd.Timedelta(seconds=15)).fillna(False)
                    sorted_group['segment_id'] = is_large_gap.cumsum()
                    seg_sizes = sorted_group['segment_id'].value_counts()
                    main_segment_id = seg_sizes.idxmax()
                    keep_indices = sorted_group[sorted_group['segment_id'] == main_segment_id].index.tolist()
                    keep_rows.extend(keep_indices)
                df = df.loc[keep_rows].drop(columns=['assigned_center', 'dt'], errors='ignore')

        self.df = df

        # Bin edges
        dEW = 0.5
        dNS = max(args.lat_grid, 0.1)
        delta_lat = dNS / self.meters_per_deg_lat
        half_delta_lon = dEW / self.meters_per_deg_lon

        bin_edges_lon = [self.lon_min]
        for center in cluster_centers:
            bin_edges_lon.extend([center - half_delta_lon, center + half_delta_lon])
        bin_edges_lon.append(self.lon_max)
        bin_edges_lon = sorted(set(bin_edges_lon))

        height_m = (self.lat_max - self.lat_min) * self.meters_per_deg_lat
        if dNS > height_m / 2 and height_m > 0:
            bin_edges_lat = np.array([self.lat_min, self.lat_max + 1e-10])
        else:
            bin_edges_lat = np.arange(self.lat_min, self.lat_max + delta_lat, delta_lat)

        # Grouping
        gps_grouped = df.groupby(['latitude', 'longitude']).agg({'frame_id': 'count'}).reset_index().rename(columns={'frame_id': 'count'})
        gps_lats = gps_grouped['latitude'].values
        gps_lons = gps_grouped['longitude'].values
        norm_gps_lats = (gps_lats - self.lat_min) / (self.lat_max - self.lat_min + 1e-12)
        norm_gps_lons = (gps_lons - self.lon_min) / (self.lon_max - self.lon_min + 1e-12)

        self.raw_grouped = df.groupby(['latitude', 'longitude'])['frame_id'].agg(lambda x: sorted(list(set(x)))).reset_index()

        if args.mode == 'filtered':
            df['lat_bin'] = pd.cut(df['latitude'], bins=bin_edges_lat, labels=range(len(bin_edges_lat)-1))
            df['lon_bin'] = pd.cut(df['longitude'], bins=bin_edges_lon, labels=range(len(bin_edges_lon)-1))
            self.grouped = df.groupby(['lat_bin', 'lon_bin'], observed=True)['frame_id'].agg(lambda x: sorted(list(set(x)))).reset_index()

            def get_bin_center(bin_id, edges):
                if pd.isna(bin_id):
                    return np.nan
                bin_id = int(bin_id)
                return (edges[bin_id] + edges[bin_id + 1]) / 2

            self.grouped['center_lat'] = self.grouped['lat_bin'].apply(get_bin_center, edges=bin_edges_lat)
            self.grouped['center_lon'] = self.grouped['lon_bin'].apply(get_bin_center, edges=bin_edges_lon)
            self.grouped = self.grouped.dropna(subset=['center_lat', 'center_lon'])
            self.grouped = self.grouped[self.grouped['frame_id'].map(len) > 0]

            self.norm_bin_edges_lat = (bin_edges_lat - self.lat_min) / (self.lat_max - self.lat_min + 1e-12)
            self.norm_bin_edges_lon = (bin_edges_lon - self.lon_min) / (self.lon_max - self.lon_min + 1e-12)

        # Plot
        self.ax.set_title("Field Map - Click to View Nearby Photos")
        self.ax.set_xlabel("Normalized Longitude")
        self.ax.set_ylabel("Normalized Latitude")

        self.scatter = self.ax.scatter(norm_gps_lons, norm_gps_lats,
                                       s=10, c=gps_grouped['count'].values, cmap='viridis', alpha=0.8,
                                       edgecolors='white', linewidth=0.5, picker=args.mode == 'raw')

        if args.mode == 'filtered':
            for edge in self.norm_bin_edges_lat:
                self.ax.axhline(edge, color='gray', ls='--', alpha=0.3)
            for edge in self.norm_bin_edges_lon:
                self.ax.axvline(edge, color='gray', ls='--', alpha=0.3)

            self.rects = []
            for _, row in self.grouped.iterrows():
                lat_bin_id = int(row['lat_bin'])
                lon_bin_id = int(row['lon_bin'])
                ll_y = self.norm_bin_edges_lat[lat_bin_id]
                ll_x = self.norm_bin_edges_lon[lon_bin_id]
                height = self.norm_bin_edges_lat[lat_bin_id + 1] - ll_y
                width = self.norm_bin_edges_lon[lon_bin_id + 1] - ll_x
                rect = Rectangle((ll_x, ll_y), width, height, facecolor='none', edgecolor='none', alpha=0, picker=True)
                rect.frame_ids = row['frame_id']
                rect.center_lat = row['center_lat']
                rect.center_lon = row['center_lon']
                self.ax.add_patch(rect)
                self.rects.append(rect)

        # Annotate columns
        norm_left = (left_most_center - self.lon_min) / (self.lon_max - self.lon_min + 1e-12)
        norm_right = (right_most_center - self.lon_min) / (self.lon_max - self.lon_min + 1e-12)
        self.ax.axvline(norm_left, color='red', ls='--', alpha=0.5)
        self.ax.axvline(norm_right, color='red', ls='--', alpha=0.5)
        self.ax.annotate(f'{left_most_center:.7f}', xy=(norm_left, 1), xycoords='axes fraction',
                         xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom', rotation=90, fontsize=9)
        self.ax.annotate(f'{right_most_center:.7f}', xy=(norm_right, 1), xycoords='axes fraction',
                         xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', rotation=90, fontsize=9)

        # Plot flowers if exists
        if os.path.exists(flowers_path):
            flowers_df = pd.read_csv(flowers_path)
            if 'latitude' in flowers_df.columns and 'longitude' in flowers_df.columns:
                flower_lats = flowers_df['latitude'].values
                flower_lons = flowers_df['longitude'].values
                norm_flower_lats = (flower_lats - self.lat_min) / (self.lat_max - self.lat_min + 1e-12)
                norm_flower_lons = (flower_lons - self.lon_min) / (self.lon_max - self.lon_min + 1e-12)
                self.flower_scatter = self.ax.scatter(norm_flower_lons, norm_flower_lats, c='red', marker='x', s=20, label='Flowers')
                self.ax.legend()

        self.canvas.draw()

        # Connect events
        if args.mode == 'raw':
            self.cid_pick = self.canvas.mpl_connect('pick_event', self.on_pick)
        elif args.mode == 'filtered':
            self.cid_pick = self.canvas.mpl_connect('pick_event', self.on_pick)
            self.cid_hover = self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:  # Left click
            return

        # Get clicked normalized coords
        norm_lon = event.xdata
        norm_lat = event.ydata

        # Convert to real
        real_lon = self.lon_min + norm_lon * (self.lon_max - self.lon_min)
        real_lat = self.lat_min + norm_lat * (self.lat_max - self.lat_min)

        # Compute distances
        dlat = self.df['latitude'] - real_lat
        dlon = self.df['longitude'] - real_lon
        dist_m = np.sqrt((dlat * self.meters_per_deg_lat)**2 + (dlon * self.meters_per_deg_lon)**2)

        # Find nearby
        nearby = self.df[dist_m < self.radius.get()]
        frame_ids = sorted(nearby['frame_id'].unique())

        if frame_ids:
            self.show_images(frame_ids, f"Photos near Lat {real_lat:.7f} | Lon {real_lon:.7f}")
        else:
            print("No photos within radius.")

    def on_pick(self, event):
        # Keep if needed for cell/point click, but radius is main
        pass

    def on_hover(self, event):
        # Preload as before if filtered
        if args.mode != 'filtered' or event.xdata is None or event.ydata is None:
            return
        # ... (same as before)

    def show_images(self, frame_ids, title):
        # Same as before
        print(f'Loading {len(frame_ids)} combined photos...')
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1440x700")

        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        scrollable_frame.grid_columnconfigure(0, weight=1)

        row = 0
        for frame_id in frame_ids:
            path = os.path.join(self.combined_folder, f"{frame_id}_original.jpg")
            label_text = f"Frame: {frame_id}"
            ttk.Label(scrollable_frame, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=0, pady=2, sticky="n")
            row += 1

            if os.path.exists(path):
                try:
                    if path in self.image_cache:
                        photo = self.image_cache[path]
                    else:
                        img = Image.open(path)
                        h = int(1440 * img.height / img.width)
                        img = img.resize((1440, h), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self.image_cache[path] = photo
                    lbl = ttk.Label(scrollable_frame, image=photo)
                    lbl.image = photo
                    lbl.grid(row=row, column=0, pady=8, sticky="n")
                except Exception as e:
                    ttk.Label(scrollable_frame, text=f"Error: {e}").grid(row=row, column=0)
                row += 1
            else:
                ttk.Label(scrollable_frame, text=f"Missing: {frame_id}_original.jpg").grid(row=row, column=0)
                row += 1

if __name__ == "__main__":
    app = GeneralImageMap()
    app.mainloop()
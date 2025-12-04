import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import ast
import math
import numpy as np
from collections import defaultdict
from PIL import Image, ImageTk
import shutil
import os
import time

# Default folder paths (edit as needed)
DEFAULT_CSV_FOLDER = r"D:\CSVs"
DEFAULT_IMAGE_FOLDER = r"D:\boxed"

# Temp folder settings
NUM_TEMP_FOLDERS = 5  # Number of rotating temp folders (temp_00 to temp_04)

# Plot settings
BASE_FIGURE_SIZE = 8.0  # Base size for figure dimensions (scales with aspect ratio)
BUFFER_FACTOR = 0.1  # Buffer around data range as fraction of max range
LEFT_MARGIN = 0.036  # Left subplot margin (increased for labels)
RIGHT_MARGIN = 0.995  # Right subplot margin
BOTTOM_MARGIN = 0.03  # Bottom subplot margin (increased for labels)
TOP_MARGIN = 0.97  # Top subplot margin
DOT_MARKER_SIZE = 5  # Size of red centroid dots
LINE_WIDTH = 1  # Width of connecting lines
SCATTER_MARKER_SIZE = 1  # Size of scatter points in main plot
SCATTER_ALPHA = 0.7  # Transparency of scatter points

# Thumbnail settings
THUMBNAIL_SPACING_METERS = 0.1  # Minimum spacing between thumbnails in meters (~1 cm)
THUMBNAIL_PIXEL_WIDTH = 200  # Pixel width for thumbnails (for visibility; height computed from aspect ratio)
SPIRAL_ANGLE_STEP = 0.6  # Angle increment for spiral offset (radians; golden-angle approximation)
SPIRAL_RADIUS_STEP_FACTOR = 0.5  # Radius step as fraction of spacing (scales with THUMBNAIL_SPACING_METERS)

# Earth approximation constants (for lat/lon to meters conversion)
METERS_PER_DEG_LAT = 111319.9  # Approximate meters per degree latitude

class FlowerViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flower Photo Viewer")
        self.geometry("800x600")
        
        # Use a modern theme for a clean, simple look
        style = ttk.Style(self)
        style.theme_use('clam')  # 'clam' provides a modern, flat appearance
        
        # Initialize variables
        self.csv_folder = None
        self.image_folder = None
        self.available_items = []  # List of (display_text, file_name) tuples
        self.data = None
        self.selected_items = []  # To store the selected display texts for plotting
        self.flower_locations = []  # To store flower positions with photo indices
        self.mode = tk.StringVar(value='session')  # Default to 'session' mode
        
        # Create main UI elements
        self.create_main_ui()
        
        # Attempt to load default folders
        self.load_default_csv_folder()
        self.load_default_image_folder()

        # Force UI refresh after 500ms to ensure list populates at startup
        self.after(500, self.update_items_listbox)

    def load_default_csv_folder(self):
        """
        Attempts to load the default CSV folder at startup.
        If successful and CSVs found, enables UI and updates status.
        If not, notifies user via messagebox and status update.
        """
        default_path = Path(DEFAULT_CSV_FOLDER)
        if default_path.exists():
            self.csv_folder = str(default_path)
            self.update_items_listbox()  # Scan based on current mode
            
            if self.available_items:
                self.listbox.config(state=tk.NORMAL)
                self.load_button.config(state=tk.NORMAL)
                self.update_status(f"Loaded default CSV folder: {self.csv_folder}. {len(self.available_items)} items available in {self.mode.get()} mode.")
            else:
                messagebox.showinfo("Default CSV Folder", "Default CSV folder found, but no valid CSV files detected for current mode. Please browse to select a different folder or add CSVs. You can edit DEFAULT_CSV_FOLDER in the code to match your system.")
                self.update_status("Default CSV folder loaded but no CSVs found. Browse to select another.")
        else:
            messagebox.showinfo("Default CSV Folder", "Default CSV folder not found. Please browse to select one. You can edit the DEFAULT_CSV_FOLDER variable in the code to match your system.")
            self.update_status("Default CSV folder not found. Please browse to select.")

    def load_default_image_folder(self):
        """
        Attempts to load the default image folder at startup.
        If successful, updates status.
        If not, notifies user via messagebox and status update.
        """
        default_path = Path(DEFAULT_IMAGE_FOLDER)
        if default_path.exists():
            self.image_folder = str(default_path)
            self.update_status(f"Loaded default image folder: {self.image_folder}.")
        else:
            messagebox.showinfo("Default Image Folder", "Default image folder not found. Please browse to select one. You can edit the DEFAULT_IMAGE_FOLDER variable in the code to match your system.")
            self.update_status("Default image folder not found. Please browse to select.")

    def create_main_ui(self):
        """
        Sets up the main window UI with Browse buttons for CSV and images, mode selection, status label, item selection listbox, and buttons.
        The listbox and Load button are initially disabled until a CSV folder is selected.
        Adds Plot button, initially disabled until data is loaded.
        """
        # Frame for CSV controls
        csv_control_frame = ttk.Frame(self)
        csv_control_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # CSV Browse button
        ttk.Button(csv_control_frame, text="Browse CSV Folder", command=self.browse_csv_folder).pack(side=tk.LEFT, padx=10)
        
        # Frame for image controls (another row)
        image_control_frame = ttk.Frame(self)
        image_control_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Image Browse button
        ttk.Button(image_control_frame, text="Browse Image Folder", command=self.browse_image_folder).pack(side=tk.LEFT, padx=10)
        
        # Mode selection frame
        mode_frame = ttk.Frame(self)
        mode_frame.pack(pady=10, padx=20, fill=tk.X)
        
        ttk.Label(mode_frame, text="Load Mode:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="By Day", variable=self.mode, value='day', command=self.update_items_listbox).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="By Session", variable=self.mode, value='session', command=self.update_items_listbox).pack(side=tk.LEFT, padx=10)
        
        # Status label (shared)
        self.status_label = ttk.Label(csv_control_frame, text="", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, expand=True)
        
        # Item selection section
        items_frame = ttk.Frame(self)
        items_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        ttk.Label(items_frame, text="Available Items:", font=("Arial", 12)).pack(anchor=tk.W)
        
        # Scrollable listbox for items (multi-select)
        scrollbar = ttk.Scrollbar(items_frame, orient="vertical")
        self.listbox = tk.Listbox(items_frame, yscrollcommand=scrollbar.set, selectmode='multiple', height=10, font=("Arial", 10))
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, pady=5, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(state=tk.DISABLED)  # Initially disabled
        
        # Load button
        self.load_button = ttk.Button(items_frame, text="Load Selected Items", command=self.load_data)
        self.load_button.pack(pady=10)
        self.load_button.config(state=tk.DISABLED)  # Initially disabled
        
        # Plot button (initially disabled)
        self.plot_button = ttk.Button(items_frame, text="Plot Flower Locations", command=self.plot_flowers)
        self.plot_button.pack(pady=10)
        self.plot_button.config(state=tk.DISABLED)  # Disabled until data loaded

    def browse_csv_folder(self):
        """
        Prompts the user to select the CSV folder.
        Updates the csv_folder, scans items based on mode, and refreshes the UI if a folder is selected.
        """
        # Show informative message with additional guidance
        messagebox.showinfo("Select CSV Folder", "Please select the folder directly containing your CSV files (named 'flowers_YYYYMMDD.csv' or 'flowers_YYYYMMDD_suffix.csv').\n\nIf the files are in a subfolder (e.g., 'CSVs'), select that specific subfolder, not the parent directory.")
        
        # Open folder dialog
        folder = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        if not folder:
            return
        
        self.csv_folder = folder
        self.update_items_listbox()
        
        # Enable/disable UI elements based on available items
        if self.available_items:
            self.listbox.config(state=tk.NORMAL)
            self.load_button.config(state=tk.NORMAL)
        else:
            self.listbox.config(state=tk.DISABLED)
            self.load_button.config(state=tk.DISABLED)
            messagebox.showwarning("No CSVs Found", f"No valid CSV files found in the selected folder for {self.mode.get()} mode.\n\nEnsure you selected the exact folder where the CSV files are located (not a parent folder). If files are in a subfolder like 'CSVs', select that one.")
        
        # Update UI
        self.update_idletasks()  # Force UI refresh after updating listbox
        self.update_status(f"CSV folder selected: {self.csv_folder}. {len(self.available_items)} items available in {self.mode.get()} mode.")
        
        # Clear current data when switching folders
        self.data = None
        self.selected_items = []
        self.plot_button.config(state=tk.DISABLED)

    def browse_image_folder(self):
        """
        Prompts the user to select the image folder.
        Updates the image_folder and refreshes the status if a folder is selected.
        """
        # Show informative message
        messagebox.showinfo("Select Image Folder", "Please select the folder containing the 'boxed' images.")
        
        # Open folder dialog
        folder = filedialog.askdirectory(title="Select Folder Containing Boxed Images")
        if not folder:
            return
        
        self.image_folder = folder
        self.update_status(f"Image folder selected: {self.image_folder}.")
        
    def scan_items(self):
        """
        Scans the selected CSV folder for files based on the current mode ('day' or 'session').
        For 'day': Looks for 'flowers_YYYYMMDD.csv' and extracts/formats dates.
        For 'session': Looks for 'flowers_YYYYMMDD_suffix.csv' and formats as 'YYYY-MM-DD_suffix'.
        Returns a sorted list of (display_text, file_name) tuples.
        Efficient for large directories as it uses glob.
        """
        path = Path(self.csv_folder)
        items = []
        
        if self.mode.get() == 'day':
            for file in path.glob('flowers_*.csv'):
                stem = file.stem
                if len(stem) == 16 and stem[8:].isdigit():  # Validate 'flowers_YYYYMMDD'
                    date_str = stem[8:]
                    display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    items.append((display, file.name))
        elif self.mode.get() == 'session':
            for file in path.glob('flowers_*_*.csv'):
                stem = file.stem
                parts = stem.split('_')
                if len(parts) >= 3 and parts[1].isdigit() and len(parts[1]) == 8:  # Validate 'flowers_YYYYMMDD_suffix'
                    date_str = parts[1]
                    suffix = '_'.join(parts[2:])  # Handle potential multi-part suffixes
                    display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}_{suffix}"
                    items.append((display, file.name))
        
        # Sort chronologically by the date part (first 8 chars after 'flowers_')
        items.sort(key=lambda x: x[1].split('_')[1])  # Sort by YYYYMMDD in file_name
        return items
    
    def update_items_listbox(self):
        """
        Updates the listbox with formatted available items based on current mode.
        """
        self.available_items = self.scan_items()
        self.listbox.delete(0, tk.END)
        for display, _ in self.available_items:
            self.listbox.insert(tk.END, display)
        self.update_idletasks()  # Force UI refresh after inserting items
    
    def update_status(self, message):
        """
        Updates the status label with the given message.
        """
        self.status_label.config(text=message)
    
    def load_data(self):
        """
        Loads the selected CSVs (based on mode) into a single DataFrame.
        Handles large datasets by reading CSVs one-by-one (pandas is memory-efficient for this scale).
        Updates the status label on success (no pop-up).
        Enables the plot button after successful load.
        Stores selected display texts for plotting.
        """
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one item.")
            return
        
        self.selected_items = [self.available_items[i][0] for i in selected_indices]
        selected_files = [self.available_items[i][1] for i in selected_indices]
        
        # Load CSVs efficiently
        dfs = []
        for file_name in selected_files:
            file = Path(self.csv_folder) / file_name
            # Read CSV; low_memory=False for large files with mixed types
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
        
        # Concatenate DataFrames (efficient for tens of thousands of rows)
        self.data = pd.concat(dfs, ignore_index=True)
        
        num_photos = len(self.data)
        
        # Update status (no pop-up)
        self.update_status(f"Loaded {num_photos} photos from selected {self.mode.get()}s. Ready for viewing/plotting.")
        
        # Enable buttons after data is loaded
        self.plot_button.config(state=tk.NORMAL)

    def plot_flowers(self):
        """
        Generates an interactive plot of all flower locations from the loaded data.
        Parses the 'GPSpoint' column efficiently to extract latitudes and longitudes.
        Converts lat/lon to relative meters using flat-earth approximation for small areas.
        Stores flower locations with photo indices for later selection.
        Handles tens of thousands of points using matplotlib scatter plot with numpy for efficiency.
        Opens the plot in a new window with navigation toolbar for zooming/panning.
        Includes selected items in the title.
        Sets equal aspect ratio for axes (1:1 scaling in meters).
        Optimizes initial figure size based on data aspect to reduce white space.
        Uses tight_layout with minimal padding to push elements against edges.
        """
        if self.data is None or self.data.empty:
            messagebox.showwarning("No Data", "No data loaded. Please load items first.")
            return
        
        # Clear previous flower locations
        self.flower_locations = []
        
        # Efficiently parse GPS points
        for i, row in self.data.iterrows():  # Efficient iteration over large df
            gps_str = row['GPSpoint']
            if pd.isna(gps_str):
                continue
            try:
                gps_list = ast.literal_eval(gps_str)
                for j, point_str in enumerate(gps_list):
                    cleaned = point_str.strip('[] ').strip()
                    parts = cleaned.split()
                    if len(parts) >= 2:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        self.flower_locations.append({
                            'photo_idx': i,
                            'flower_idx': j,
                            'lat': lat,
                            'lon': lon
                        })
            except (ValueError, SyntaxError):
                pass
        
        if not self.flower_locations:
            messagebox.showwarning("No Points", "No valid GPS points found in the data.")
            return
        
        # Extract lats and lons as numpy arrays
        all_lats = np.array([f['lat'] for f in self.flower_locations])
        all_lons = np.array([f['lon'] for f in self.flower_locations])
        
        # Compute average latitude for longitude scaling
        avg_lat = np.mean(all_lats)
        
        # Approximate meters per degree
        meters_per_deg_lat = METERS_PER_DEG_LAT
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(avg_lat))
        
        # Reference point
        ref_lat = np.min(all_lats)
        ref_lon = np.min(all_lons)
        
        # Convert to relative meters and store in flower_locations
        for f in self.flower_locations:
            f['y_m'] = (f['lat'] - ref_lat) * meters_per_deg_lat
            f['x_m'] = (f['lon'] - ref_lon) * meters_per_deg_lon
        
        # Numpy arrays for plotting and computations
        self.all_x_m = np.array([f['x_m'] for f in self.flower_locations])
        self.all_y_m = np.array([f['y_m'] for f in self.flower_locations])
        
        # Compute data ranges for aspect ratio
        x_min, x_max = np.min(self.all_x_m), np.max(self.all_x_m)
        y_min, y_max = np.min(self.all_y_m), np.max(self.all_y_m)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        data_aspect = x_range / y_range
        
        # Set initial figure size based on data aspect
        base_size = BASE_FIGURE_SIZE
        fig_width = base_size * data_aspect
        fig_height = base_size
        
        # Format selected items for title
        formatted_items = ', '.join(sorted(self.selected_items))
        
        # Create a new top-level window for the plot
        self.plot_win = tk.Toplevel(self)
        self.plot_win.title("Flower Locations Plot")
        self.plot_win.geometry("1200x900")
        
        # Create matplotlib figure and axis
        self.fig = plt.Figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_subplot(111)
        
        # Scatter plot
        self.ax.scatter(self.all_x_m, self.all_y_m, marker='.', s=SCATTER_MARKER_SIZE, alpha=SCATTER_ALPHA)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal', adjustable='datalim')
        
        # Set labels and title
        self.ax.set_xlabel("East-West (meters)")
        self.ax.set_ylabel("North-South (meters)")
        self.ax.set_title(f"Flower Locations ({len(self.flower_locations)} points) - Selected: {formatted_items}")
        
        # Adjust subplot margins
        self.fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, bottom=BOTTOM_MARGIN, top=TOP_MARGIN)
        
        # Embed the figure in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_win)
        self.canvas.draw()
        
        # Add navigation toolbar at the top
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_win)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Pack canvas to fill the remaining space and expand on resize
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add bottom frame for Select Circle button
        bottom_frame = ttk.Frame(self.plot_win)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Select Circle button in bottom left
        ttk.Button(bottom_frame, text="Select Circle", command=self.activate_circle_selection).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(bottom_frame, text="Select Rectangle", command=self.activate_rect_selection).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Bind <Configure> event for dynamic resizing
        def on_resize(event):
            new_width = event.width
            new_height = event.height - toolbar.winfo_height() - bottom_frame.winfo_height()
            if new_width <= 0 or new_height <= 0:
                return
            dpi = self.fig.get_dpi()
            new_fig_width = new_width / dpi
            new_fig_height = new_height / dpi
            min_size = 1.0
            new_fig_width = max(min_size, new_fig_width)
            new_fig_height = max(min_size, new_fig_height)
            current_size = self.fig.get_size_inches()
            if abs(new_fig_width - current_size[0]) > 0.01 or abs(new_fig_height - current_size[1]) > 0.01:
                self.fig.set_size_inches(new_fig_width, new_fig_height, forward=True)
                self.canvas.draw()
        
        self.plot_win.bind("<Configure>", on_resize)

        # Store reference and scaling for later use in info.txt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.meters_per_deg_lat = meters_per_deg_lat
        self.meters_per_deg_lon = meters_per_deg_lon

    def activate_circle_selection(self):
        """
        Activates the selection tool on the plot to select two points for the circle.
        """
        if self.image_folder is None:
            messagebox.showwarning("No Image Folder", "Please select an image folder first.")
            return
        
        if not hasattr(self, 'plot_win') or not self.plot_win.winfo_exists():
            messagebox.showwarning("No Plot", "Please plot the flower locations first.")
            return
        
        if not self.flower_locations:
            messagebox.showwarning("No Data", "No flower locations available.")
            return
        
        # Reset and set mode to circle
        self.selection_mode = 'circle'
        self.center = None
        self.circle = None
        self.cid_press = None
        self.cid_motion = None
        
        # Connect press event
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press_selection)
        
        # Update status
        self.update_status("Click the center of the circle, then move mouse and click on circumference.")

    def activate_rect_selection(self):
        """
        Activates the selection tool on the plot to select two diagonal corners for the rectangle.
        """
        if self.image_folder is None:
            messagebox.showwarning("No Image Folder", "Please select an image folder first.")
            return
        
        if not hasattr(self, 'plot_win') or not self.plot_win.winfo_exists():
            messagebox.showwarning("No Plot", "Please plot the flower locations first.")
            return
        
        if not self.flower_locations:
            messagebox.showwarning("No Data", "No flower locations available.")
            return
        
        # Reset and set mode to rect
        self.selection_mode = 'rect'
        self.corner1 = None
        self.rect = None
        self.cid_press = None
        self.cid_motion = None
        
        # Connect press event
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press_selection)
        
        # Update status
        self.update_status("Click one corner of the rectangle, then move mouse and click the opposite corner.")

    def on_press_selection(self, event):
        """
        Handles press events for both circle and rectangle selection.
        """
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        
        if self.selection_mode == 'circle':
            if self.center is None:
                self.center = (event.xdata, event.ydata)
                self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion_selection)
            else:
                self.canvas.mpl_disconnect(self.cid_press)
                self.canvas.mpl_disconnect(self.cid_motion)
                if self.circle:
                    self.circle.remove()
                    self.circle = None
                self.canvas.draw()
                radius = math.hypot(event.xdata - self.center[0], event.ydata - self.center[1])
                self.process_selection(self.center, radius=radius, corners=None)
        
        elif self.selection_mode == 'rect':
            if self.corner1 is None:
                self.corner1 = (event.xdata, event.ydata)
                self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion_selection)
            else:
                self.canvas.mpl_disconnect(self.cid_press)
                self.canvas.mpl_disconnect(self.cid_motion)
                if self.rect:
                    self.rect.remove()
                    self.rect = None
                self.canvas.draw()
                corner2 = (event.xdata, event.ydata)
                self.process_selection(None, radius=None, corners=(self.corner1, corner2))

    def on_motion_selection(self, event):
        """
        Handles motion events to preview the circle or rectangle.
        """
        if event.xdata is None or event.ydata is None:
            return
        
        if self.selection_mode == 'circle' and self.center:
            radius = math.hypot(event.xdata - self.center[0], event.ydata - self.center[1])
            if self.circle:
                self.circle.remove()
            self.circle = Circle(self.center, radius, fill=False, color='red', linestyle='--', linewidth=2)
            self.ax.add_patch(self.circle)
            self.canvas.draw_idle()
        
        elif self.selection_mode == 'rect' and self.corner1:
            x1, y1 = self.corner1
            x2, y2 = event.xdata, event.ydata
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x = min(x1, x2)
            y = min(y1, y2)
            if self.rect:
                self.rect.remove()
            self.rect = Rectangle((x, y), width, height, fill=False, color='red', linestyle='--', linewidth=2)
            self.ax.add_patch(self.rect)
            self.canvas.draw_idle()

    def process_selection(self, center, radius, corners):
        """
        Processes the selected area (circle or rectangle), finds images within it, and displays them in a new window with relative positions.
        Thumbnails are resized while preserving the original 1920x1080 aspect ratio (approximately 16:9).
        Thumbnail width is set to 300 pixels for a balance between visibility and performance; height is computed accordingly.
        This ensures efficient memory usage and smooth rendering even with up to 50 thumbnails in the plot,
        as each thumbnail is ~300x169 pixels (~150KB in memory), totaling ~7.5MB for 50 images.
        Numpy is used for efficient distance computations on large datasets.
        """
        # Efficiently find inside indices
        if self.selection_mode == 'circle':
            dists = np.sqrt((self.all_x_m - center[0]) ** 2 + (self.all_y_m - center[1]) ** 2)
            inside_idx = np.nonzero(dists <= radius)[0]
        else:  # rect
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            inside_idx = np.nonzero((self.all_x_m >= x_min) & (self.all_x_m <= x_max) &
                                    (self.all_y_m >= y_min) & (self.all_y_m <= y_max))[0]
        
        if len(inside_idx) == 0:
            messagebox.showinfo("No Images", "No flowers within the selected area.")
            self.update_status("No images found.")
            return
        
        # Group flowers by photo_idx using defaultdict for efficiency
        photo_flowers = defaultdict(list)
        for idx in inside_idx:
            f = self.flower_locations[idx]
            photo_flowers[f['photo_idx']].append((f['x_m'], f['y_m']))
        
        # Collect and sort selected rows by timestamp for chronological order
        selected_rows = [self.data.iloc[photo_idx] for photo_idx in photo_flowers.keys()]
        selected_rows.sort(key=lambda row: row['timestamp'])
        
        # Compute data ranges for selected points
        selected_x = self.all_x_m[inside_idx]
        selected_y = self.all_y_m[inside_idx]
        x_min, x_max = np.min(selected_x), np.max(selected_x)
        y_min, y_max = np.min(selected_y), np.max(selected_y)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        data_aspect = x_range / y_range
        
        # Set initial figure size based on data aspect
        base_size = BASE_FIGURE_SIZE
        fig_width = base_size * data_aspect
        fig_height = base_size
        
        # Create new window for images
        self.image_win = tk.Toplevel(self)
        self.image_win.title("Images in Selected Area")
        self.image_win.geometry("1200x900")
        
        # Create matplotlib figure and axis
        self.image_fig = plt.Figure(figsize=(fig_width, fig_height))
        self.image_ax = self.image_fig.add_subplot(111)
        
        # Set limits with buffer
        buffer = BUFFER_FACTOR * max(x_range, y_range)
        self.image_ax.set_xlim(x_min - buffer, x_max + buffer)
        self.image_ax.set_ylim(y_min - buffer, y_max + buffer)
        self.image_ax.set_aspect('equal', adjustable='datalim')
        
        # Add the selection patch
        if self.selection_mode == 'circle':
            self.image_ax.add_patch(Circle(center, radius, fill=False, color='red', linewidth=2))
        else:
            width = x_max - x_min
            height = y_max - y_min
            self.image_ax.add_patch(Rectangle((x_min, y_min), width, height, fill=False, color='red', linewidth=2))
        
        # Set labels and title
        self.image_ax.set_xlabel("East-West (meters)")
        self.image_ax.set_ylabel("North-South (meters)")
        
        # Load and place images at centroids, with spread for overlaps
        loaded_count = 0
        # Define thumbnail size preserving 1920x1080 aspect ratio (16:9)
        # Width=300 for good visibility; height computed to maintain ratio
        # This reduces data size per thumbnail for efficient handling of up to 50 images
        thumb_width = THUMBNAIL_PIXEL_WIDTH
        thumb_height = round(thumb_width * 1080 / 1920)  # ≈169
        thumb_size = (thumb_width, thumb_height)
        self.image_artists = []  # To store artists and their paths for clicking
        self.lines = []  # To store connecting lines
        self.dots = []  # To store centroid dots
        positions = []  # To track thumbnail positions for overlap check
        for photo_idx, pos_list in photo_flowers.items():
            pos_arr = np.array(pos_list)
            centroid_x = np.mean(pos_arr[:, 0])
            centroid_y = np.mean(pos_arr[:, 1])
            
            row = self.data.iloc[photo_idx]
            image_path = Path(self.image_folder) / f"{row['timestamp'].rstrip('Z')}_{row['camera']}.jpg"
            if not image_path.exists():
                continue  # Skip missing images
            
            try:
                # Load and resize image efficiently with PIL, preserving aspect ratio
                img_pil = Image.open(image_path).resize(thumb_size, Image.Resampling.LANCZOS)
                img_arr = np.array(img_pil)
                # Initial position at centroid
                pos_x, pos_y = centroid_x, centroid_y
                # Check for overlap with existing positions (efficient vectorized distance)
                if positions:
                    dists = np.sqrt((np.array(positions) - np.array([pos_x, pos_y]))**2).sum(axis=1)
                    if np.any(dists < THUMBNAIL_SPACING_METERS):
                        # Offset in spiral pattern (efficient incremental search)
                        angle = 0
                        r = THUMBNAIL_SPACING_METERS   # Start with threshold as radius
                        while True:
                            offset_x = r * math.cos(angle)
                            offset_y = r * math.sin(angle)
                            test_x = centroid_x + offset_x
                            test_y = centroid_y + offset_y
                            dists = np.sqrt((np.array(positions) - np.array([test_x, test_y]))**2).sum(axis=1)
                            if np.all(dists >= THUMBNAIL_SPACING_METERS):
                                pos_x, pos_y = test_x, test_y
                                break
                            angle += SPIRAL_ANGLE_STEP
                            r += THUMBNAIL_SPACING_METERS * SPIRAL_RADIUS_STEP_FACTOR   # Scale step with threshold
                positions.append((pos_x, pos_y))
                imagebox = OffsetImage(img_arr, zoom=1)
                ab = AnnotationBbox(imagebox, (pos_x, pos_y), frameon=False)
                self.image_ax.add_artist(ab)
                self.image_artists.append((ab, str(image_path)))
                loaded_count += 1
                # Add dot at true centroid
                dot, = self.image_ax.plot(centroid_x, centroid_y, 'ro', markersize=DOT_MARKER_SIZE)
                self.dots.append(dot)
                # Add line from centroid to thumbnail position
                line, = self.image_ax.plot([centroid_x, pos_x], [centroid_y, pos_y], 'k-', linewidth=LINE_WIDTH)
                self.lines.append(line)
            except Exception:
                continue  # Skip invalid images
        
        self.image_ax.set_title(f"Images in Selected Area ({loaded_count} images)")
        
        if loaded_count == 0:
            messagebox.showinfo("No Images", "No valid images found within the area.")
            self.image_win.destroy()
            self.update_status("No images found.")
            return
        
        # Adjust subplot margins
        self.image_fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, bottom=BOTTOM_MARGIN, top=TOP_MARGIN)
        
        # Embed in tkinter
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=self.image_win)
        self.image_canvas.draw()
        
        # Add navigation toolbar at the top
        toolbar = NavigationToolbar2Tk(self.image_canvas, self.image_win)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Pack canvas to fill the remaining space and expand on resize
        self.image_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click event for opening full image
        self.image_canvas.mpl_connect('button_press_event', self.on_image_click)
        
        # Enable dragging of thumbnails
        self.dragged_ab = None
        self.drag_index = None
        self.cid_pick = self.image_canvas.mpl_connect('pick_event', self.on_pick_thumbnail)
        self.cid_release = self.image_canvas.mpl_connect('button_release_event', self.on_release_thumbnail)
        self.cid_motion = self.image_canvas.mpl_connect('motion_notify_event', self.on_drag_thumbnail)
        
        # Bind <Configure> event for dynamic resizing
        def on_image_resize(event):
            new_width = event.width
            new_height = event.height - toolbar.winfo_height()
            if new_width <= 0 or new_height <= 0:
                return
            dpi = self.image_fig.get_dpi()
            new_fig_width = new_width / dpi
            new_fig_height = new_height / dpi
            min_size = 1.0
            new_fig_width = max(min_size, new_fig_width)
            new_fig_height = max(min_size, new_fig_height)
            current_size = self.image_fig.get_size_inches()
            if abs(new_fig_width - current_size[0]) > 0.01 or abs(new_fig_height - current_size[1]) > 0.01:
                self.image_fig.set_size_inches(new_fig_width, new_fig_height, forward=True)
                self.image_canvas.draw()
        
        self.image_win.bind("<Configure>", on_image_resize)
        
        # Handle full-resolution images in temp folder
        temp_folder = self.get_temp_folder()
        if temp_folder.exists():
            shutil.rmtree(temp_folder)  # Clear previous contents efficiently
        temp_folder.mkdir(parents=True, exist_ok=True)
        
        # Extract temp number from folder name (e.g., '00' from 'temp_00')
        temp_num = temp_folder.name[-2:]
        
        # Create subfolders for cameras
        rs1_folder = temp_folder / f'{temp_num}_RS1'
        rs2_folder = temp_folder / f'{temp_num}_RS2'
        rs3_folder = temp_folder / f'{temp_num}_RS3'
        rs1_folder.mkdir(exist_ok=True)
        rs2_folder.mkdir(exist_ok=True)
        rs3_folder.mkdir(exist_ok=True)
        
        # Copy images with original filenames to main folder and subfolders grouped by camera
        for row in selected_rows:
            src_path = Path(self.image_folder) / f"{row['timestamp'].rstrip('Z')}_{row['camera']}.jpg"
            if src_path.exists():
                # Copy to main temp folder
                dst_main = temp_folder / src_path.name
                shutil.copy2(src_path, dst_main)  # Preserve metadata for efficiency
                
                # Copy to camera subfolder
                camera = row['camera']
                if camera == 'RS1':
                    dst_sub = rs1_folder / src_path.name
                elif camera == 'RS2':
                    dst_sub = rs2_folder / src_path.name
                elif camera == 'RS3':
                    dst_sub = rs3_folder / src_path.name
                else:
                    continue  # Skip if unknown camera
                shutil.copy2(src_path, dst_sub)
        
        # Compute center GPS and meters
        if self.selection_mode == 'circle':
            center_x_m, center_y_m = center
            center_lat = self.ref_lat + center_y_m / self.meters_per_deg_lat
            center_lon = self.ref_lon + center_x_m / self.meters_per_deg_lon
            selection_type = "Circle"
            additional_info = f"Radius: {radius:.2f} meters\n"
        else:  # rect
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            center_x_m = (x1 + x2) / 2
            center_y_m = (y1 + y2) / 2
            center_lat = self.ref_lat + center_y_m / self.meters_per_deg_lat
            center_lon = self.ref_lon + center_x_m / self.meters_per_deg_lon
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            selection_type = "Rectangle"
            additional_info = f"Width: {width:.2f} meters\nHeight: {height:.2f} meters\n"
        
        # Write info.txt with numbered name
        info_path = temp_folder / f'{temp_num}_info.txt'
        with info_path.open('w') as f:
            f.write(f"Selection Type: {selection_type}\n")
            f.write(f"Center GPS: lat={center_lat:.8f}, lon={center_lon:.8f}\n")
            f.write(f"Center Meters: x={center_x_m:.2f} (East-West), y={center_y_m:.2f} (North-South)\n")
            f.write(additional_info)
            f.write(f"Number of Images: {loaded_count}\n")
            f.write(f"Sessions: {', '.join(self.selected_items)}\n")
        
        # Open the folder with Windows Photos (or default app)
        os.startfile(str(temp_folder))
        
        self.update_status(f"Displayed {loaded_count} images. Full views in {temp_folder.name} (opened in Photos app).")

    def on_pick_thumbnail(self, event):
        """
        Handles picking a thumbnail for dragging.
        """
        # Check if an AnnotationBbox was picked
        if isinstance(event.artist, AnnotationBbox):
            self.dragged_ab = event.artist
            self.drag_start = (event.mouseevent.xdata, event.mouseevent.ydata)
            # Find index to update the correct line
            self.drag_index = next(i for i, (ab, _) in enumerate(self.image_artists) if ab == self.dragged_ab)

    def on_drag_thumbnail(self, event):
        """
        Handles dragging the thumbnail and updating the connecting line.
        """
        if self.dragged_ab and event.xdata and event.ydata:
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
            current_xy = list(self.dragged_ab.xy)
            new_xy = (current_xy[0] + dx, current_xy[1] + dy)
            self.dragged_ab.xy = new_xy
            # Update line
            line = self.lines[self.drag_index]
            dot = self.dots[self.drag_index]
            line.set_data([dot.get_xdata()[0], new_xy[0]], [dot.get_ydata()[0], new_xy[1]])
            self.drag_start = (event.xdata, event.ydata)
            self.image_canvas.draw_idle()

    def on_release_thumbnail(self, event):
        """
        Handles release after dragging a thumbnail.
        """
        self.dragged_ab = None

    def get_temp_folder(self):
        """
        Selects the temp_XX folder to reuse: prefers non-existent, then oldest by modification time.
        Efficient scan of only NUM_TEMP_FOLDERS possible folders.
        """
        parent = Path(self.csv_folder).parent  # One level up from CSVs folder
        candidates = []
        for i in range(NUM_TEMP_FOLDERS):
            folder = parent / f'temp_{i:02d}'
            if folder.exists():
                mtime = folder.stat().st_mtime
            else:
                mtime = float('-inf')  # Prefer non-existent
            candidates.append((mtime, folder))
        
        # Sort by mtime (oldest first)
        candidates.sort()
        return candidates[0][1]  # Return the selected folder path

    def on_image_click(self, event):
        """
        Handles clicks on the image plot to open full-size image if clicked on a thumbnail.
        """
        if event.button != 1 or event.inaxes is None:
            return
        
        for ab, image_path in self.image_artists:
            contains, _ = ab.contains(event)
            if contains:
                self.open_full_image(image_path)
                return

    def open_full_image(self, image_path):
        """
        Opens the full-size image using the system's default photo viewer app.
        On Windows, this launches the file with the registered default program (e.g., Photos app),
        which handles high-resolution viewing, zooming, and navigation efficiently without loading
        the image into Python memory—ideal for large datasets of sensor images in robotic mapping.
        """
        import os  # Import here for modularity; os.startfile is Windows-specific but efficient
        
        try:
            os.startfile(image_path)  # Simple, zero-overhead call to open with default app
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image with default viewer: {e}")

if __name__ == "__main__":
    app = FlowerViewer()
    app.mainloop()
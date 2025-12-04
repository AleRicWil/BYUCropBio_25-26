import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import ast
import math
import numpy as np
from collections import defaultdict
from PIL import Image, ImageTk

# Default CSV folder path
# Edit this to match the default location on your system
DEFAULT_CSV_FOLDER = r"D:\CSVs"

# Default image folder path
# Edit this to match the default location on your system
DEFAULT_IMAGE_FOLDER = r"D:\boxed"

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
        self.available_dates = []
        self.data = None
        self.selected_dates = []  # To store the loaded dates for plotting
        self.flower_locations = []  # To store flower positions with photo indices
        
        # Create main UI elements
        self.create_main_ui()
        
        # Attempt to load default folders
        self.load_default_csv_folder()
        self.load_default_image_folder()

    def load_default_csv_folder(self):
        """
        Attempts to load the default CSV folder at startup.
        If successful and CSVs found, enables UI and updates status.
        If not, notifies user via messagebox and status update.
        """
        default_path = Path(DEFAULT_CSV_FOLDER)
        if default_path.exists():
            self.csv_folder = str(default_path)
            self.available_dates = self.scan_dates()
            
            if self.available_dates:
                self.listbox.config(state=tk.NORMAL)
                self.load_button.config(state=tk.NORMAL)
                self.update_dates_listbox()
                self.update_status(f"Loaded default CSV folder: {self.csv_folder}. {len(self.available_dates)} days available.")
            else:
                messagebox.showinfo("Default CSV Folder", "Default CSV folder found, but no valid 'flowers_YYYYMMDD.csv' files detected. Please browse to select a different folder or add CSVs. You can edit DEFAULT_CSV_FOLDER in the code to match your system.")
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
        Sets up the main window UI with Browse buttons for CSV and images, status label, day selection listbox, and buttons.
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
        
        # Status label (shared)
        self.status_label = ttk.Label(csv_control_frame, text="", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, expand=True)
        
        # Day selection section
        days_frame = ttk.Frame(self)
        days_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        ttk.Label(days_frame, text="Available Days:", font=("Arial", 12)).pack(anchor=tk.W)
        
        # Scrollable listbox for dates (multi-select)
        scrollbar = ttk.Scrollbar(days_frame, orient="vertical")
        self.listbox = tk.Listbox(days_frame, yscrollcommand=scrollbar.set, selectmode='multiple', height=10, font=("Arial", 10))
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, pady=5, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(state=tk.DISABLED)  # Initially disabled
        
        # Load button
        self.load_button = ttk.Button(days_frame, text="Load Selected Days", command=self.load_data)
        self.load_button.pack(pady=10)
        self.load_button.config(state=tk.DISABLED)  # Initially disabled
        
        # Plot button (initially disabled)
        self.plot_button = ttk.Button(days_frame, text="Plot Flower Locations", command=self.plot_flowers)
        self.plot_button.pack(pady=10)
        self.plot_button.config(state=tk.DISABLED)  # Disabled until data loaded

    def browse_csv_folder(self):
        """
        Prompts the user to select the CSV folder.
        Updates the csv_folder, scans dates, and refreshes the UI if a folder is selected.
        """
        # Show informative message with additional guidance
        messagebox.showinfo("Select CSV Folder", "Please select the folder directly containing your daily CSV files (named 'flowers_YYYYMMDD.csv').\n\nIf the files are in a subfolder (e.g., 'CSVs'), select that specific subfolder, not the parent directory.")
        
        # Open folder dialog
        folder = filedialog.askdirectory(title="Select Folder Containing Daily CSV Files")
        if not folder:
            return  # Canceled
        
        self.csv_folder = folder
        self.available_dates = self.scan_dates()
        
        # Enable/disable UI elements based on available dates
        if self.available_dates:
            self.listbox.config(state=tk.NORMAL)
            self.load_button.config(state=tk.NORMAL)
        else:
            self.listbox.config(state=tk.DISABLED)
            self.load_button.config(state=tk.DISABLED)
            messagebox.showwarning("No CSVs Found", "No valid 'flowers_YYYYMMDD.csv' files found in the selected folder.\n\nEnsure you selected the exact folder where the CSV files are located (not a parent folder). If files are in a subfolder like 'CSVs', select that one.")
        
        # Update UI
        self.update_dates_listbox()
        self.update_idletasks()  # Force UI refresh after updating listbox
        self.update_status(f"CSV folder selected: {self.csv_folder}. {len(self.available_dates)} days available.")
        
        # Clear current data when switching folders
        self.data = None
        self.selected_dates = []
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
            return  # Canceled
        
        self.image_folder = folder
        self.update_status(f"Image folder selected: {self.image_folder}.")
        
    def scan_dates(self):
        """
        Scans the selected CSV folder for files named 'flowers_YYYYMMDD.csv' and extracts the dates.
        Returns a sorted list of date strings (YYYYMMDD).
        Efficient for large directories as it uses glob.
        """
        path = Path(self.csv_folder)
        dates = []
        for file in path.glob('flowers_*.csv'):
            date_str = file.stem[8:]  # Extract YYYYMMDD from 'flowers_YYYYMMDD'
            if len(date_str) == 8 and date_str.isdigit():  # Basic validation
                dates.append(date_str)
        dates.sort()  # Sort chronologically
        return dates
    
    def update_dates_listbox(self):
        """
        Updates the listbox with formatted available dates.
        """
        self.listbox.delete(0, tk.END)
        for date in self.available_dates:
            # Format date for display: YYYY-MM-DD
            formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            self.listbox.insert(tk.END, formatted)
        self.update_idletasks()  # Force UI refresh after inserting items
    
    def update_status(self, message):
        """
        Updates the status label with the given message.
        """
        self.status_label.config(text=message)
    
    def load_data(self):
        """
        Loads the selected daily CSVs into a single DataFrame.
        Handles large datasets by reading CSVs one-by-one (pandas is memory-efficient for this scale).
        Updates the status label on success (no pop-up).
        Enables the plot button after successful load.
        Stores selected dates for plotting.
        """
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one day.")
            return
        
        self.selected_dates = [self.available_dates[i] for i in selected_indices]
        
        # Load CSVs efficiently
        dfs = []
        for date in self.selected_dates:
            file = Path(self.csv_folder) / f'flowers_{date}.csv'
            # Read CSV; low_memory=False for large files with mixed types
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
        
        # Concatenate DataFrames (efficient for tens of thousands of rows)
        self.data = pd.concat(dfs, ignore_index=True)
        
        num_photos = len(self.data)
        
        # Update status (no pop-up)
        self.update_status(f"Loaded {num_photos} photos. Ready for viewing/plotting.")
        
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
        Includes MM/DD of loaded days in the title.
        Sets equal aspect ratio for axes (1:1 scaling in meters).
        Optimizes initial figure size based on data aspect to reduce white space.
        Uses tight_layout with minimal padding to push elements against edges.
        """
        if self.data is None or self.data.empty:
            messagebox.showwarning("No Data", "No data loaded. Please load days first.")
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
        meters_per_deg_lat = 111319.9
        meters_per_deg_lon = 111319.9 * math.cos(math.radians(avg_lat))
        
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
        base_size = 8.0
        fig_width = base_size * data_aspect
        fig_height = base_size
        
        # Format selected dates as MM/DD
        formatted_days = ', '.join(f"{date[4:6]}/{date[6:]}" for date in sorted(self.selected_dates))
        
        # Create a new top-level window for the plot
        self.plot_win = tk.Toplevel(self)
        self.plot_win.title("Flower Locations Plot")
        self.plot_win.geometry("1200x900")
        
        # Create matplotlib figure and axis
        self.fig = plt.Figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_subplot(111)
        
        # Scatter plot
        self.ax.scatter(self.all_x_m, self.all_y_m, marker='.', s=1, alpha=0.7)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal', adjustable='datalim')
        
        # Set labels and title
        self.ax.set_xlabel("East-West (meters)")
        self.ax.set_ylabel("North-South (meters)")
        self.ax.set_title(f"Flower Locations ({len(self.flower_locations)} points) - Days: {formatted_days}")
        
        # Adjust subplot margins
        self.fig.subplots_adjust(left=0.026, right=0.995, bottom=0.02, top=0.97)
        
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
        ttk.Button(bottom_frame, text="Select Circle", command=self.activate_selection).pack(side=tk.LEFT, padx=10, pady=10)
        
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

    def activate_selection(self):
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
        
        # Reset circle and connections
        self.center = None
        self.circle = None
        self.cid_press = None
        self.cid_motion = None
        
        # Connect press event
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        
        # Update status
        self.update_status("Click the center of the circle, then move mouse and click on circumference.")

    def on_press(self, event):
        """
        Handles press events for circle definition.
        """
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        
        if self.center is None:
            # First click: set center
            self.center = (event.xdata, event.ydata)
            # Connect motion event
            self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        else:
            # Second click: finalize
            self.canvas.mpl_disconnect(self.cid_press)
            self.canvas.mpl_disconnect(self.cid_motion)
            if self.circle:
                self.circle.remove()
                self.circle = None
            self.canvas.draw()
            self.process_circle(self.center, event)

    def on_motion(self, event):
        """
        Handles motion events to preview the circle.
        """
        if self.center is None or event.xdata is None or event.ydata is None:
            return
        
        radius = math.hypot(event.xdata - self.center[0], event.ydata - self.center[1])
        
        if self.circle:
            self.circle.remove()
        
        self.circle = Circle(self.center, radius, fill=False, color='red', linestyle='--', linewidth=2)
        self.ax.add_patch(self.circle)
        self.canvas.draw_idle()

    def process_circle(self, center, event):
        """
        Processes the selected circle, finds images within it, and displays them in a new window with relative positions.
        Thumbnails are resized while preserving the original 1920x1080 aspect ratio (approximately 16:9).
        Thumbnail width is set to 300 pixels for a balance between visibility and performance; height is computed accordingly.
        This ensures efficient memory usage and smooth rendering even with up to 50 thumbnails in the plot,
        as each thumbnail is ~300x169 pixels (~150KB in memory), totaling ~7.5MB for 50 images.
        Numpy is used for efficient distance computations on large datasets.
        """
        radius = math.hypot(event.xdata - center[0], event.ydata - center[1])
        
        # Compute distances efficiently with numpy for large point sets
        dists = np.sqrt((self.all_x_m - center[0]) ** 2 + (self.all_y_m - center[1]) ** 2)
        inside_idx = np.nonzero(dists <= radius)[0]
        
        if len(inside_idx) == 0:
            messagebox.showinfo("No Images", "No flowers within the selected circle.")
            self.update_status("No images found.")
            return
        
        # Group flowers by photo_idx using defaultdict for efficiency
        photo_flowers = defaultdict(list)
        for idx in inside_idx:
            f = self.flower_locations[idx]
            photo_flowers[f['photo_idx']].append((f['x_m'], f['y_m']))
        
        # Compute data ranges for selected points
        selected_x = self.all_x_m[inside_idx]
        selected_y = self.all_y_m[inside_idx]
        x_min, x_max = np.min(selected_x), np.max(selected_x)
        y_min, y_max = np.min(selected_y), np.max(selected_y)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        data_aspect = x_range / y_range
        
        # Set initial figure size based on data aspect
        base_size = 8.0
        fig_width = base_size * data_aspect
        fig_height = base_size
        
        # Create new window for images
        self.image_win = tk.Toplevel(self)
        self.image_win.title("Images in Circle")
        self.image_win.geometry("1200x900")
        
        # Create figure and axis
        self.image_fig = plt.Figure(figsize=(fig_width, fig_height))
        self.image_ax = self.image_fig.add_subplot(111)
        
        # Set limits with buffer
        buffer = max(x_range, y_range) * 0.1
        self.image_ax.set_xlim(x_min - buffer, x_max + buffer)
        self.image_ax.set_ylim(y_min - buffer, y_max + buffer)
        self.image_ax.set_aspect('equal', adjustable='datalim')
        
        # Add the selection circle
        self.image_ax.add_patch(Circle(center, radius, fill=False, color='red', linewidth=2))
        
        # Set labels and title
        self.image_ax.set_xlabel("East-West (meters)")
        self.image_ax.set_ylabel("North-South (meters)")
        
        # Load and place images at centroids
        loaded_count = 0
        # Define thumbnail size preserving 1920x1080 aspect ratio (16:9)
        # Width=300 for good visibility; height computed to maintain ratio
        # This reduces data size per thumbnail for efficient handling of up to 50 images
        thumb_width = 300
        thumb_height = round(thumb_width * 1080 / 1920)  # ≈169
        thumb_size = (thumb_width, thumb_height)
        self.image_artists = []  # To store artists and their paths for clicking
        for photo_idx, positions in photo_flowers.items():
            pos_arr = np.array(positions)
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
                imagebox = OffsetImage(img_arr, zoom=1)
                ab = AnnotationBbox(imagebox, (centroid_x, centroid_y), frameon=False)
                self.image_ax.add_artist(ab)
                self.image_artists.append((ab, str(image_path)))
                loaded_count += 1
            except Exception:
                continue  # Skip invalid images
        
        self.image_ax.set_title(f"Images in Selected Circle ({loaded_count} images)")
        
        if loaded_count == 0:
            messagebox.showinfo("No Images", "No valid images found within the circle.")
            self.image_win.destroy()
            self.update_status("No images found.")
            return
        
        # Adjust subplot margins
        self.image_fig.subplots_adjust(left=0.026, right=0.995, bottom=0.02, top=0.97)
        
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
        
        self.update_status(f"Displayed {loaded_count} images.")

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
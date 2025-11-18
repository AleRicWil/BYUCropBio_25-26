import cv2  # pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Variable parameters
filename = r'LED Tip Tracking\Two Straights\01.mov'
fps = 60
myNum_leds = 3
myROI_size = 50
myFilter_vals = 255, 255, 255, 30  # R, G, B, tolerance values for color filter
show_frames = False # step frame by frame during detection to troubleshoot detection
show_video = True  # show simple video during detection to quickly verify detection
show_paths = True  # show results of detection and use GUI to measure angles

def mouse_callback(event, x, y, flags, param):
    '''Helper for user to select LED starting position'''
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def led_detection(frame, roi_center, roi_size=myROI_size, filter_vals=None, show_frames=False):
    """
    Computes the centroid of bright pixels in the ROI (region of interest) after grayscale thresholding. ROI is
    centered on previous frame's LED location.
    Input: frame (BGR color image), roi_center (x, y), roi_size (square side length), filter_vals (r,g,b,tol).
    Output: (x_global, y_global, led_radius) or None if no centroid is found.
    """
    r, g, b, tol = filter_vals
    threshold = 255 - tol  # Threshold for grayscale based on tolerance
    
    # Extract ROI from BGR frame
    x_c, y_c = roi_center
    half_size = roi_size // 2
    x_min = max(0, x_c - half_size)
    x_max = min(frame.shape[1], x_c + half_size)
    y_min = max(0, y_c - half_size)
    y_max = min(frame.shape[0], y_c + half_size)
    roi_bgr = frame[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    
    # Threshold to create binary image
    _, binary = cv2.threshold(roi_blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours to get size of bright spot. Not required
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Select the largest contour by area
        led_contour = max(contours, key=cv2.contourArea)
        _, led_radius = cv2.minEnclosingCircle(led_contour)
        led_radius = int(led_radius)
    else:
        led_radius = 0
    
    # Compute moments to find centroid
    moments = cv2.moments(binary)
    if moments['m00'] != 0:
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        x_global = x + x_min
        y_global = y + y_min
        
        # Frame by frame troubleshooting
        if show_frames:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(roi_gray, cmap='gray')
            ax[0].set_title("ROI Grayscale")
            ax[1].imshow(binary, cmap='gray')
            ax[1].set_title("Binary Image")
            if led_radius > 0:
                ax[1].add_patch(plt.Circle((x, y), led_radius, color='green', fill=False))
            plt.show()
        return (x_global, y_global, led_radius)
    return None

def track_tips(cap, mouse_callback, filter_vals=None, num_leds=1, display_video=False):
    '''
    For a given video capture, tracks multiple LEDs in each frame and returns their pixel locations
    '''
    # Video info and variable setup
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time = np.arange(0, num_frames/fps, 1/fps)
    print('Frames:', num_frames); print(f'Select {num_leds} LEDs')
    positions = [np.full((num_frames, 3), np.nan) for _ in range(num_leds)]
    prev_centers = [None] * num_leds
    tip_velocities = [None] * num_leds
    
    # Loop to find pixel location of each LED in each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # User selects starting point of each LED
        if frame_idx == 0:
            global click_pos
            # Get frame dimensions
            h, w = frame.shape[:2]
            # Define a maximum display size while preserving aspect ratio
            max_display_size = 800  # Maximum width or height
            scale = min(max_display_size / w, max_display_size / h)
            display_w, display_h = int(w * scale), int(h * scale)
            
            for led_idx in range(num_leds):
                click_pos = None
                cv2.namedWindow("Click Rod Tip", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Click Rod Tip", display_w, display_h)
                cv2.setMouseCallback("Click Rod Tip", mouse_callback)
                while click_pos is None:
                    # Resize frame for display
                    display_frame = cv2.resize(frame, (display_w, display_h))
                    cv2.imshow("Click Rod Tip", display_frame)
                    cv2.setWindowTitle("Click Rod Tip", f"Click Rod Tip {led_idx + 1}/{num_leds}")
                    if cv2.waitKey(10) & 0xFF == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                # Adjust click position back to original frame size
                x_click = int(click_pos[0] / scale)
                y_click = int(click_pos[1] / scale)
                prev_centers[led_idx] = (x_click, y_click)
                cv2.circle(frame, (x_click, y_click), 3, (0, 0, 255), -1)  # Red for initial click
            print('Tracking LEDs...')
            if not display_video:
                cv2.destroyAllWindows()

        # Find the location of each LED in the frame
        for led_idx in range(num_leds):
            if (tip_velocities[led_idx] is None) or (tip_velocities[led_idx] <= myROI_size*0.2):
                this_ROI_size = myROI_size
            else:
                print('here')
                this_ROI_size = int(myROI_size * 2)
            result = led_detection(frame, prev_centers[led_idx], roi_size=this_ROI_size, filter_vals=filter_vals, show_frames=show_frames)
            if result is not None:
                x, y, r = result
                positions[led_idx][frame_idx] = [x, y, r]
                x_prev, y_prev = prev_centers[led_idx]
                pixels_traveled = np.sqrt((x-x_prev)**2 + (y-y_prev)**2)
                tip_velocities[led_idx] = pixels_traveled
                prev_centers[led_idx] = (x, y)
                if display_video:
                    x_display = int(x)
                    y_display = int(y)
                    cv2.circle(frame, (x_display, y_display), 3, (0, 0, 255), -1)  # Red for detected position
        if display_video:
            cv2.imshow("LED Detection", frame)
            cv2.waitKey(1)
        
        # Move to next frame for next cycle of loop
        frame_idx += 1

    if display_video:
        cv2.destroyAllWindows()
    
    return time, positions

def play_and_select_brightness(cap, mouse_callback):
    """
    Use for selecting color filters for the video. Ideally only the LED gets through the filter.
    Plays a video and allows brightness-based color filtering using sliders.
    Get good values, then manually enter them in header section
    Input: cap (cv2.VideoCapture object), mouse_callback (function for mouse click events).
    Output: None.
    """
    frame_idx = 0
    root = tk.Tk()
    root.title("RGB Color Filter")
    r_value = tk.DoubleVar(value=255)
    g_value = tk.DoubleVar(value=255)
    b_value = tk.DoubleVar(value=255)
    tolerance = tk.DoubleVar(value=50)
    running = tk.BooleanVar(value=True)

    def update_image(frame):
        if not running.get():
            return frame
        r = int(r_value.get())
        g = int(g_value.get())
        b = int(b_value.get())
        tol = int(tolerance.get())
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lower_bound = np.array([max(0, r - tol), max(0, g - tol), max(0, b - tol)])
        upper_bound = np.array([min(255, r + tol), min(255, g + tol), min(255, b + tol)])
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        filtered_img = img_rgb.copy()
        filtered_img[mask == 0] = [0, 0, 0]
        return cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR)

    def stop_video():
        running.set(False)
        root.destroy()

    ttk.Label(root, text="Red (0-255):").pack()
    ttk.Scale(root, from_=0, to=255, variable=r_value).pack()
    ttk.Label(root, text="Green (0-255):").pack()
    ttk.Scale(root, from_=0, to=255, variable=g_value).pack()
    ttk.Label(root, text="Blue (0-255):").pack()
    ttk.Scale(root, from_=0, to=255, variable=b_value).pack()
    ttk.Label(root, text="Tolerance (0-100):").pack()
    ttk.Scale(root, from_=0, to=100, variable=tolerance).pack()
    ttk.Button(root, text="Stop", command=stop_video).pack()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        filtered_frame = update_image(frame)
        cv2.imshow("Filtered Video", filtered_frame)
        root.update()
        if cv2.waitKey(1) & 0xFF == 27 or not running.get():
            break
        print('Frame:', frame_idx)
        frame_idx += 1

def playback_paths(positions):
    '''
    After detection has run, this presents the results in an interactive window. Allows angle measurements
    to be taken by pausing video with 'Space', which are displayed in a readout window. 
    '''
    cap2 = cv2.VideoCapture(filename)
    num_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap2.isOpened():
        print("Error opening video for playback")
        return
    
    print('Playback with tip paths\nPress "Space" to pause')
    # Variables for pause state, mouse clicks, and lines
    paused = False
    click_points = []  # List to store two clicked points during pause
    lines = []  # List to store all lines [(pt1, pt2), ...]
    line_colors = []  # Store colors for each line
    color_idx = [0]  # Index for cycling through colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Predefined colors
    
    # Mouse callback to capture clicks during pause
    def mouse_callback2(event, x, y, flags, param):
        if paused and event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            if len(click_points) == 2:  # Once two points are clicked, store the line
                lines.append((click_points[0], click_points[1]))
                line_colors.append(colors[color_idx[0] % len(colors)])  # Assign next color
                color_idx[0] += 1
                click_points.clear()  # Reset for the next line
                update_display()
    
    # Function to update both windows for each frame's data
    def update_display():
        frame_copy = frame.copy()
        # Draw full path for all LEDs up to the current frame
        for led_idx in range(myNum_leds):
            for pos_idx in range(min(frame_idx + 1, len(positions[led_idx]))):
                pos = positions[led_idx][pos_idx]
                if not np.isnan(pos[0]):
                    x, y = int(pos[0]), int(pos[1])
                    if pos_idx >= 1:
                        cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)  # Red dot
                    else:
                        size = 30
                        cv2.line(frame_copy, (x - size, y), (x + size, y), (0, 255, 255), 3)
                        cv2.line(frame_copy, (x, y - size), (x, y + size), (0, 255, 255), 3)
        
        # Draw all stored lines with their colors
        for (pt1, pt2), color in zip(lines, line_colors):
            cv2.line(frame_copy, pt1, pt2, color, 2)
        
        # Draw clicked points during pause (if any)
        for pt in click_points:
            cv2.circle(frame_copy, pt, 5, (0, 255, 0), -1)  # Green dot for clicked points
        
        cv2.imshow("LED Path Playback", frame_copy)

        '''End video window'''
        '''Start angle readout window'''

        readout_text = ['Angles']
        for i, ((x1, y1), (x2, y2)) in enumerate(lines):
            dx = x2 - x1
            dy = y2 - y1
            angle = -np.degrees(np.arctan2(dy, dx))# % 360
            # if angle < 0:
            #     angle += 360
            # angle = min(angle, 360 - angle)  # Acute angle from right horizontal
            color_str = f"({line_colors[i][2]}, {line_colors[i][1]}, {line_colors[i][0]})"  # BGR to RGB order for display
            readout_text.append(f"Line {i+1}: Angle = {angle:.1f}, Color = {color_str}")
        
        # Create a scrollable text image
        text_height = 30
        num_lines = len(readout_text)
        window_height = min(500, text_height * num_lines)  # Max height of 500 pixels
        readout_img = np.zeros((window_height, 800, 3), dtype=np.uint8) # 800 pixels wide
        readout_img.fill(255)  # White background
        
        scroll_offset = 0
        if num_lines * text_height > window_height:
            scroll_offset = max(0, min(scroll_offset, (num_lines * text_height) - window_height))
        
        for i, line in enumerate(readout_text):
            y = i * text_height - scroll_offset
            if 0 <= y < window_height:
                cv2.putText(readout_img, line, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        cv2.imshow("Line Angles", readout_img)
    
    # Set up the mouse callback with 1:1 pixel mapping
    cv2.namedWindow("LED Path Playback", cv2.WINDOW_NORMAL)
    # Get frame dimensions (do this after reading the first frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap2.read()
    if ret:
        h, w = frame.shape[:2]
        cv2.resizeWindow("LED Path Playback", w, h)  # Set window size to match frame
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    cv2.setMouseCallback("LED Path Playback", mouse_callback2)
    
    '''Main loop for running interactive window'''
    frame_idx = 0
    delay_ms = int(1000 / fps)  # Delay per frame to match real-time playback
    while cap2.isOpened():
        # Read frame based on frame_idx
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap2.read()
        if not ret or frame_idx >= len(positions[0]):
            break
        
        update_display()
        
        # Handle key presses
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key == ord(' '):  # Space to toggle pause/resume
            paused = not paused
            if paused:
                print("Paused - Click two points to draw a line, or use 'a' and 'd' to step")
            else:
                click_points.clear()  # Clear any incomplete points
                print("Resumed")
        elif paused and key == ord('d'):  # 'd' to step forward
            frame_idx = min(frame_idx + 1, len(positions[0]) - 1)
            print(f"Stepped forward to frame {frame_idx}")
        elif paused and key == ord('a'):  # 'a' to step backward
            frame_idx = max(0, frame_idx - 1)
            print(f"Stepped back to frame {frame_idx}")
        
        if not paused:
            frame_idx += 1
            if frame_idx >= num_frames - 2:
                paused = not paused
    
    # Keep window open after video finishes until 'q' or 'Esc' is pressed
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Esc
            break
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cap1 = cv2.VideoCapture(filename)
    if not cap1.isOpened():
        print("Error opening video")
        exit()

    # Uncomment to use for selecting color filters. Only needed once per video if default values don't work
    # play_and_select_brightness(cap1, mouse_callback)

    # Set display_video=True to watch the video with detected positions marked by red dots
    time, myPositions = track_tips(cap1, mouse_callback, myFilter_vals, myNum_leds, display_video=show_video)
    cap1.release()
    cv2.destroyAllWindows()

    # Results
    # for i in range(myNum_leds):
    #     data = np.column_stack((time, myPositions[i]))
    #     np.savetxt(f'Tip_Tracking_Results_May27_4_{i}.txt', data, header='Time(s),X,Y,Radius', comments='', fmt='%.0f')
    #     plt.figure(10 + i)
    #     plt.scatter(myPositions[i][:, 0], myPositions[i][:, 1])
    if show_paths:
        playback_paths(myPositions)
        
    plt.show()
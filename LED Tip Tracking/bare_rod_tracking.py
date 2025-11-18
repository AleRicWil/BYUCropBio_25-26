import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

filename = r'rod_test.mp4'
writeFile = r'results.txt'
# Mouse callback function to capture click position
# click_pos = None
def mouse_callback(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def custom_circle_detection(gradient_magnitude, gray_frame, roi_center, roi_size=200):
    """
    Custom circle detection within an ROI.
    Input: gradient_magnitude, gray_frame, roi_center (x, y), roi_size (square side length).
    Output: (x, y, r) or None if no circle detected.
    """
    # Define ROI boundaries
    x_c, y_c = roi_center
    half_size = roi_size // 2
    x_min = max(0, x_c - half_size)
    x_max = min(gray_frame.shape[1], x_c + half_size)
    y_min = max(0, y_c - half_size)
    y_max = min(gray_frame.shape[0], y_c + half_size)
    
    # Extract ROI from gradient_magnitude and gray_frame
    roi_grad = gradient_magnitude[y_min:y_max, x_min:x_max]
    roi_gray = gray_frame[y_min:y_max, x_min:x_max]
    # Preprocess roi_gray: enhance contrast and reduce noise
    roi_gray = cv2.equalizeHist(roi_gray)
    roi_gray = cv2.GaussianBlur(roi_gray, (11, 11), 0)

    # Find circles in image region of interest
    circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                param1=70, param2=25, minRadius=15, maxRadius=30)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Choose circle closest to previous center
        if len(circles) > 1 and prev_center is not None:
            distances = [np.sqrt((x - half_size)**2 + (y - half_size)**2) for x, y, r in circles]
            closest_idx = np.argmin(distances)
            x, y, r = circles[closest_idx]
        else:
            x, y, r = circles[0]
        print('Circles:', len(circles))

        
        # Show all circles found
        # fig2, ax2 = plt.subplots(2,1)
        # ax2[0].imshow(roi_grad, cmap='gray')
        # ax2[0].set_title(f'Radius: {r}')
        # ax2[0].scatter(half_size, half_size, s=5, c='red')
        # ax2[1].imshow(roi_gray, cmap='gray')
        # ax2[1].scatter(half_size, half_size, s=5, c='red')
        # for x_i, y_i, r_i, in circles:
        #     ax2[1].scatter(x_i, y_i, s=5, c='green')
        #     ax2[1].add_patch(plt.Circle((x_i, y_i), r_i, color='green', fill=False))
        # ax2[1].scatter(x, y, s=5, c='orange')
        # ax2[1].add_patch(plt.Circle((x, y), r, color='orange', fill=False))
        # plt.show()

        # Adjust coordinates to global frame
        x_global = x + x_min
        y_global = y + y_min
        return (x_global, y_global, r)
    return None


# Load the video

cap = cv2.VideoCapture(filename)
if not cap.isOpened():
    print("Error opening video")
    exit()

# Get the total number of frames
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print('Frames:', num_frames)

# Initialize positions array to store (x, y, r) for each frame
positions = np.full((num_frames, 3), np.nan)  # x, y, radius

# Run gradient-based detection to populate positions
frame_idx = 0
prev_center = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Compute gradients using Sobel filters
    grad_x = ndimage.sobel(blurred, axis=1)
    grad_y = ndimage.sobel(blurred, axis=0)
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize to [0, 255] for display
    gradient_magnitude_display = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    # Resize images to half their original size for display
    target_width = gray.shape[1] // 2
    target_height = gray.shape[0] // 2
    gray_resized = cv2.resize(blurred, (target_width, target_height))
    grad_resized = cv2.resize(gradient_magnitude_display, (target_width, target_height))
    # Convert resized images to BGR for consistent display
    gray_display = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    grad_display = cv2.cvtColor(grad_resized, cv2.COLOR_GRAY2BGR)
    # Stack images horizontally for display (used after first frame)
    combined_display = np.hstack((gray_display, grad_display))
    
    # For first frame, get user input via mouse click on grayscale only
    if frame_idx == 0:
        global click_pos
        click_pos = None
        cv2.namedWindow("Click Rod Tip (Grayscale)")
        cv2.setMouseCallback("Click Rod Tip (Grayscale)", mouse_callback)
        while click_pos is None:
            cv2.imshow("Click Rod Tip (Grayscale)", gray_display)
            if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                exit()
        # Adjust click position to original frame size
        x_click = int(click_pos[0] * (gray.shape[1] / target_width))
        y_click = int(click_pos[1] * (gray.shape[0] / target_height))
        prev_center = (x_click, y_click)
    
    
    
    # Run custom circle detection with ROI centered on prev_center
    result = custom_circle_detection(gradient_magnitude, blurred, prev_center)
    if result is not None:
        x, y, r = result
        positions[frame_idx] = [x, y, r]
        prev_center = (x, y)  # Update center for next frame
        # Adjust coordinates for resized display (half-size) and draw on grayscale (left side)
        x_display = int(x * (target_width / gray.shape[1]))
        y_display = int(y * (target_height / gray.shape[0]))
        # Draw a small red circle (radius 3) on the grayscale portion of combined_display
        cv2.circle(combined_display, (x_display, y_display), 3, (0, 0, 255), -1)
    # Display detection inputs (combined grayscale and gradient magnitude)
    cv2.imshow("Detection Input (Left: Grayscale, Right: Gradient Magnitude)", combined_display)
    cv2.waitKey(1)
    
    print('Frame:', frame_idx); frame_idx += 1

# cv2.destroyWindow("Detection Input (Left: Grayscale, Right: Gradient Magnitude)")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Results
np.savetxt(writeFile, positions)
plt.figure(10)
plt.scatter(positions[:, 0], positions[:, 1])
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_sparse_optical_flow(prev_frame, curr_frame, max_corners=100):
    """
    Extract sparse optical flow using Lucas-Kanade method.
    
    Args:
        prev_frame: Previous grayscale frame
        curr_frame: Current grayscale frame
        max_corners: Maximum number of corners to track
        
    Returns:
        prev_points: Points in previous frame
        next_points: Corresponding points in current frame
        flow_visualization: Visualization of the optical flow
        avg_motion: Average motion vector [dx, dy]
    """
    # Parameters for corner detection
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Detect corners in the first frame
    prev_points = cv2.goodFeaturesToTrack(prev_frame, mask=None, **feature_params)
    
    # Default return values if no points are found
    avg_motion = np.zeros(2)
    
    if prev_points is None:
        return None, None, curr_frame, avg_motion
    
    # Calculate optical flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, curr_frame, prev_points, None, **lk_params
    )
    
    # Only keep good points where status == 1
    good_old = prev_points[status == 1]
    good_new = next_points[status == 1]
    
    # Create visualization
    # Convert current frame to BGR if it's grayscale
    if len(curr_frame.shape) == 2:
        flow_visualization = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
    else:
        flow_visualization = curr_frame.copy()
    
    # Calculate average motion
    if len(good_new) > 0 and len(good_old) > 0:
        # Calculate motion vectors
        motion_vectors = good_new - good_old
        
        # Use median to be robust against outliers
        if len(motion_vectors) >= 5:
            avg_motion = np.median(motion_vectors, axis=0)
        else:
            avg_motion = np.mean(motion_vectors, axis=0)
    
    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Current point
        c, d = old.ravel()  # Previous point
        
        # Convert to int for drawing
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Draw line showing motion
        cv2.line(flow_visualization, (a, b), (c, d), (0, 255, 0), 2)
        # Draw circle at new position
        cv2.circle(flow_visualization, (a, b), 5, (0, 0, 255), -1)
    
    return good_old, good_new, flow_visualization, avg_motion

def create_trajectory_image(position_history, image_size=(600, 600)):
    """Create an image showing the trajectory"""
    traj_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Set the center of the image as the starting point
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    
    # Scale factor for better visualization
    scale = 1.0  # Adjust as needed
    
    # Draw coordinate system
    cv2.line(traj_image, (center_x, center_y), (center_x + 50, center_y), (0, 0, 255), 2)  # X-axis
    cv2.line(traj_image, (center_x, center_y), (center_x, center_y - 50), (0, 255, 0), 2)  # Y-axis
    
    # Draw trajectory points
    prev_x, prev_y = center_x, center_y
    
    for i, pos in enumerate(position_history):
        # Calculate pixel position
        x = int(center_x + pos[0] * scale)
        y = int(center_y + pos[1] * scale)
        
        # Ensure within bounds
        x = max(0, min(image_size[0]-1, x))
        y = max(0, min(image_size[1]-1, y))
        
        # Draw point
        cv2.circle(traj_image, (x, y), 1, (255, 255, 255), -1)
        
        # Draw line connecting to previous point (except for first point)
        if i > 0:
            cv2.line(traj_image, (prev_x, prev_y), (x, y), (0, 255, 255), 1)
        
        prev_x, prev_y = x, y
    
    # Draw current position with larger marker
    if position_history:
        current_x = int(center_x + position_history[-1][0] * scale)
        current_y = int(center_y + position_history[-1][1] * scale)
        current_x = max(0, min(image_size[0]-1, current_x))
        current_y = max(0, min(image_size[1]-1, current_y))
        cv2.circle(traj_image, (current_x, current_y), 5, (0, 255, 0), -1)
    
    # Add text showing current position
    if position_history:
        pos_text = f"Position: ({position_history[-1][0]:.2f}, {position_history[-1][1]:.2f})"
        cv2.putText(traj_image, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return traj_image

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or file path for video
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        return
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_points = None
    
    # For trajectory tracking
    position = np.zeros(2)  # [x, y] position
    position_history = []  # List to store position history
    
    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # For the first frame or if we need to reset
        if prev_points is None:
            # Parameters for corner detection
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            # Create visualization for first frame
            vis = frame.copy()
            # Show initial points
            if prev_points is not None:
                for point in prev_points:
                    x, y = point.ravel()
                    cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            avg_motion = np.zeros(2)
        else:
            # Calculate sparse optical flow
            prev_points, next_points, vis, avg_motion = extract_sparse_optical_flow(prev_gray, gray)
            
            # Update points for next iteration
            if next_points is not None and len(next_points) > 10:
                prev_points = next_points.reshape(-1, 1, 2)
            else:
                # Reset if we lost too many points
                prev_points = None
        
        # Update position (negate motion since camera moving right means scene moving left)
        position[0] -= avg_motion[0]
        position[1] -= avg_motion[1]
        
        # Store position history
        position_history.append(position.copy())
        
        # Display optical flow
        cv2.imshow('Sparse Optical Flow', vis)
        
        # Display trajectory
        trajectory_image = create_trajectory_image(position_history)
        cv2.imshow('Trajectory', trajectory_image)
        
        # Update previous frame
        prev_gray = gray
        
        # Break loop with 'q' key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

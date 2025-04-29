import cv2
import numpy as np
import time
from collections import deque


class VisualOdometry:
    def __init__(
        self,
        max_corners=500,
        quality_level=0.3,
        min_distance=7,
        block_size=7,
        win_size=(100, 100),
        max_level=8,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        motion_scale=0.2,
    ):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=criteria,
        )

        self.motion_scale = motion_scale
        self.prev_gray = None
        self.prev_points = None
        self.position = np.zeros(2, dtype=float)

    def reset_features(self, gray_frame):
        """Reinitialize feature points on the given grayscale frame."""
        self.prev_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)

    def process(self, frame):
        """
        Process a frame to update the position estimate.

        Args:
            frame (np.ndarray): BGR or grayscale image.

        Returns:
            np.ndarray: Current x,y position.
            np.ndarray: Last computed velocity vector.
            dict: Additional visualization data
        """
        # Ensure grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First-time initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            self.reset_features(gray)
            return self.position.copy(), np.zeros(2), {'points': None}

        velocity = np.zeros(2)
        viz_data = {'points': None}
        
        if self.prev_points is None or len(self.prev_points) == 0:
            self.reset_features(gray)
        else:
            # Track features
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            if next_pts is not None and status is not None:
                # Flatten points to shape (N,2)
                mask = status.flatten() == 1
                good_old = self.prev_points[mask].reshape(-1, 2)
                good_new = next_pts[mask].reshape(-1, 2)

                # Store points for visualization
                viz_data['points'] = (good_old, good_new)

                if good_new.size and good_old.size:
                    motion = good_new - good_old
                    # Compute robust velocity
                    vel = (
                        np.median(motion, axis=0)
                        if len(motion) >= 5
                        else np.mean(motion, axis=0)
                    )
                    # Negative because camera moves opposite to scene motion
                    velocity = vel
                    self.position -= velocity * self.motion_scale

                # Refresh or continue tracking
                if len(good_new) > 10:
                    self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    self.reset_features(gray)
            else:
                self.reset_features(gray)

        self.prev_gray = gray
        return self.position.copy(), velocity, viz_data


def main():
    cap = cv2.VideoCapture(0)
    vo = VisualOdometry(motion_scale=0.2)

    # Optional: tailor capture properties
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Path visualization
    path_length = 500  # Store last N positions
    path = deque(maxlen=path_length)
    
    # For map visualization
    map_size = 400
    map_scale = 2  # Pixels per unit of movement
    map_center = (map_size // 2, map_size // 2)
    
    # For performance tracking
    start_time = time.time()
    prev_time = start_time
    fps_history = deque(maxlen=30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            current_time = time.time()
            position, velocity, viz_data = vo.process(frame)
            dt = current_time - prev_time
            fps_history.append(1/dt if dt > 0 else 0)
            prev_time = current_time

            # Reset position if less than 3 seconds have passed (stabilization)
            if current_time - start_time < 3:
                vo.position = np.zeros(2)
                path.clear()
            
            # Add current position to path
            path.append(position.copy())

            # Create visualization output
            output_frame = frame.copy()
            
            # Draw optical flow
            if viz_data['points'] is not None:
                good_old, good_new = viz_data['points']
                for i, (old, new) in enumerate(zip(good_old, good_new)):
                    a, b = old.astype(int)
                    c, d = new.astype(int)
                    # Draw flow lines
                    cv2.line(output_frame, (a, b), (c, d), (0, 255, 0), 2)
                    # Draw current points
                    cv2.circle(output_frame, (c, d), 3, (0, 0, 255), -1)
            
            # Create trajectory map
            map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
            
            # Draw grid lines
            grid_step = 20
            for i in range(0, map_size, grid_step):
                cv2.line(map_img, (0, i), (map_size-1, i), (220, 220, 220), 1)
                cv2.line(map_img, (i, 0), (i, map_size-1), (220, 220, 220), 1)
            
            # Draw center cross
            cv2.line(map_img, (map_center[0], map_center[1]-10), (map_center[0], map_center[1]+10), (0, 0, 0), 2)
            cv2.line(map_img, (map_center[0]-10, map_center[1]), (map_center[0]+10, map_center[1]), (0, 0, 0), 2)
            
            # Draw path
            if len(path) > 1:
                for i in range(1, len(path)):
                    pt1 = (
                        int(map_center[0] + path[i-1][0] * map_scale),
                        int(map_center[1] + path[i-1][1] * map_scale)
                    )
                    pt2 = (
                        int(map_center[0] + path[i][0] * map_scale),
                        int(map_center[1] + path[i][1] * map_scale)
                    )
                    # Draw path with gradient color (older=blue, newer=red)
                    color_factor = i / len(path)
                    color = (
                        int(255 * (1 - color_factor)),  # Blue component
                        0,                              # Green component
                        int(255 * color_factor)         # Red component
                    )
                    cv2.line(map_img, pt1, pt2, color, 2)
            
            # Draw current position marker
            current_pos = (
                int(map_center[0] + position[0] * map_scale),
                int(map_center[1] + position[1] * map_scale)
            )
            cv2.circle(map_img, current_pos, 5, (0, 0, 255), -1)
            
            # Display position and velocity text
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            info_text = [
                f"Position: ({position[0]:.2f}, {position[1]:.2f})",
                f"Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f})",
                f"FPS: {avg_fps:.1f}"
            ]
            
            # Resize map to fit in display
            map_display = cv2.resize(map_img, (output_frame.shape[1] // 3, output_frame.shape[0] // 3))
            
            # Overlay map on bottom right
            h, w = map_display.shape[:2]
            output_frame[output_frame.shape[0]-h:, output_frame.shape[1]-w:] = map_display
            
            # Add text information
            for i, text in enumerate(info_text):
                cv2.putText(
                    output_frame, text, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            # Show output
            cv2.imshow("Visual Odometry", output_frame)
            
            # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

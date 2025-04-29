import cv2
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanVisualOdometry:
    def __init__(
        self,
        max_corners=100,
        quality_level=0.3,
        min_distance=7,
        block_size=7,
        win_size=(15, 15),  # Reduced window size for faster processing
        max_level=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        motion_scale=0.2,
        process_noise=0.1,  # Increased to be more responsive to changes
        measurement_noise=0.2,
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
        
        # FilterPy Kalman filter initialization
        self.kf = None
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.last_time = time.time()
        self.initialized = False
        self.dt = 1.0  # Initial dt value
        
        # For responsiveness tracking
        self.processing_times = []
        self.max_processing_history = 30
        
        # For outlier rejection
        self.max_velocity = 50.0  # Maximum allowable pixel velocity

    def _init_kalman_filter(self):
        """Initialize the Kalman Filter with current dt value"""
        # State: [x, y, vx, vy]
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise covariance
        kf.R = np.eye(2) * self.measurement_noise
        
        # Process noise covariance
        q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.process_noise)
        kf.Q = np.zeros((4, 4))
        kf.Q[:2, :2] = q * 0.05  # Lower process noise for position
        kf.Q[2:, 2:] = q * 2.0   # Higher process noise for velocity to be more responsive
        
        # Initial state uncertainty
        kf.P = np.eye(4) * 100  # Start with high uncertainty
        
        # Initial state (will be set on first measurement)
        kf.x = np.zeros((4, 1))
        
        return kf

    def reset_features(self, gray_frame):
        """Reinitialize feature points on the given grayscale frame."""
        # Use a mask to focus on the center region for more stable tracking
        h, w = gray_frame.shape
        mask = np.zeros_like(gray_frame)
        mask[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 255
        
        self.prev_points = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)

    def process(self, frame):
        """
        Process a frame to update the position estimate.
        """
        start_time = time.time()
        
        # Resize frame for faster processing if needed
        # frame = cv2.resize(frame, (320, 240))
        
        # Ensure grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dt (time since last frame)
        current_time = time.time()
        self.dt = min(current_time - self.last_time, 0.1)  # Cap dt to avoid jumps
        self.last_time = current_time
        
        # Initialize Kalman filter if needed
        if self.kf is None:
            self.kf = self._init_kalman_filter()

        # First-time initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            self.reset_features(gray)
            return self.position.copy(), np.zeros(2)

        # Kalman filter prediction step (with updated dt)
        if self.initialized:
            # Update state transition matrix with current dt
            self.kf.F[0, 2] = self.dt
            self.kf.F[1, 3] = self.dt
            
            # Update process noise with current dt
            q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.process_noise)
            self.kf.Q[:2, :2] = q * 0.05
            self.kf.Q[2:, 2:] = q * 2.0
            
            # Predict next state
            self.kf.predict()

        raw_velocity = np.zeros(2)
        measured_position = self.position.copy()
        measurement_valid = False
        
        # Calculate optical flow
        if self.prev_points is None or len(self.prev_points) < 10:
            self.reset_features(gray)
        else:
            # Track features with optical flow - core of responsiveness
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            if next_pts is not None and status is not None:
                # Flatten points to shape (N,2)
                mask = status.flatten() == 1
                good_old = self.prev_points[mask].reshape(-1, 2)
                good_new = next_pts[mask].reshape(-1, 2)

                if len(good_new) >= 5 and len(good_old) >= 5:
                    # Calculate motion vectors
                    motion = good_new - good_old
                    
                    # Reject outliers for more stable tracking
                    motion_magnitudes = np.linalg.norm(motion, axis=1)
                    median_magnitude = np.median(motion_magnitudes)
                    
                    # Keep only motions within reasonable range of median
                    inlier_mask = motion_magnitudes < (median_magnitude * 2 + 0.1)
                    if np.sum(inlier_mask) >= 5:
                        good_motion = motion[inlier_mask]
                        
                        # Compute robust velocity using median
                        vel = np.median(good_motion, axis=0)
                        
                        # Check if velocity is within reasonable bounds
                        vel_magnitude = np.linalg.norm(vel)
                        if vel_magnitude < self.max_velocity:
                            # Negative because camera moves opposite to scene motion
                            raw_velocity = vel
                            
                            # Calculate position delta from velocity
                            delta_position = -raw_velocity * self.motion_scale
                            measured_position = self.position + delta_position
                            measurement_valid = True
                            
                            # Update Kalman filter with measurement
                            if not self.initialized:
                                # First valid measurement - initialize the filter
                                self.kf.x = np.array([[measured_position[0]], 
                                                    [measured_position[1]], 
                                                    [-raw_velocity[0] * self.motion_scale / max(self.dt, 0.001)], 
                                                    [-raw_velocity[1] * self.motion_scale / max(self.dt, 0.001)]])
                                self.initialized = True
                            else:
                                # Update with measurement, using adaptive noise
                                # More uncertain measurements (high velocity) get higher noise
                                noise_scale = max(1.0, min(vel_magnitude / 10.0, 5.0))
                                self.kf.R = np.eye(2) * self.measurement_noise * noise_scale
                                self.kf.update(measured_position)
                            
                            # Get filtered state
                            self.position = self.kf.x[:2, 0]

                # Refresh or continue tracking
                if len(good_new) > 10:
                    self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    self.reset_features(gray)
            else:
                self.reset_features(gray)

        self.prev_gray = gray.copy()  # Make explicit copy to avoid reference issues
        
        # If we got no valid measurement but filter is initialized, use predicted position
        if not measurement_valid and self.initialized:
            self.position = self.kf.x[:2, 0]
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_history:
            self.processing_times.pop(0)
        
        # Return Kalman-filtered position and velocity
        if self.initialized:
            return self.position.copy(), self.kf.x[2:4, 0]
        else:
            return self.position.copy(), np.zeros(2)
            
    def get_covariance(self):
        """Return the position covariance from the Kalman filter"""
        if self.initialized:
            return self.kf.P[:2, :2]
        return np.eye(2) * 999  # High uncertainty if not initialized
        
    def get_avg_processing_time(self):
        """Return average processing time per frame"""
        if not self.processing_times:
            return 0
        return np.mean(self.processing_times)


def main():
    cap = cv2.VideoCapture(0)
    vo = KalmanVisualOdometry(
        motion_scale=0.2,
        process_noise=0.1,    # Higher for responsiveness
        measurement_noise=0.2,
        win_size=(15, 15),    # Smaller for faster processing
        max_corners=80,       # Fewer corners for speed
    )

    # Optional: tailor capture properties for performance
    cap.set(cv2.CAP_PROP_FPS, 30)  # Lower FPS for more stable processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Smaller frame for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # For visualization
    path_img = np.zeros((480, 480, 3), dtype=np.uint8)
    center = np.array([path_img.shape[1]//2, path_img.shape[0]//2])
    
    # Colors for visualization
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    
    frame_count = 0
    total_time = 0
    fps = 0

    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            position, velocity = vo.process(frame)
            
            # Get position covariance (uncertainty)
            pos_cov = vo.get_covariance()
            uncertainty = np.sqrt(np.diag(pos_cov))  # Standard deviation
            
            # Draw trajectory on path image (with fading effect)
            path_img = np.maximum(path_img - 1, 0)  # Fade older points
            
            point = center + (position * 20).astype(int)  # Scale for visualization
            # Ensure point is within bounds
            point[0] = max(0, min(point[0], path_img.shape[1] - 1))
            point[1] = max(0, min(point[1], path_img.shape[0] - 1))
            
            # Draw position with uncertainty ellipse
            cv2.circle(path_img, tuple(point), 2, GREEN, -1)
            
            # Draw velocity vector
            vel_endpoint = point + (velocity * 10).astype(int)
            # Ensure endpoint is within bounds
            vel_endpoint[0] = max(0, min(vel_endpoint[0], path_img.shape[1] - 1))
            vel_endpoint[1] = max(0, min(vel_endpoint[1], path_img.shape[0] - 1))
            cv2.arrowedLine(path_img, tuple(point), tuple(vel_endpoint), BLUE, 2)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / total_time
                frame_count = 0
                total_time = 0
            
            # Display information on frame
            cv2.putText(frame, f"Pos: ({position[0]:.1f}, {position[1]:.1f})", 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            cv2.putText(frame, f"Vel: ({velocity[0]:.1f}, {velocity[1]:.1f})", 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Process: {vo.get_avg_processing_time()*1000:.1f}ms", 
                      (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display
            cv2.imshow('Camera View', frame)
            cv2.imshow('Filtered Trajectory', path_img)

            # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            # Track loop timing
            loop_time = time.time() - loop_start
            total_time += loop_time
            
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

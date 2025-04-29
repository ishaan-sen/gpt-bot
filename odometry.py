import cv2
import numpy as np
import time


class VisualOdometry:
    def __init__(
        self,
        max_corners=100,
        quality_level=0.3,
        min_distance=7,
        block_size=7,
        win_size=(30, 30),
        max_level=2,
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
        """
        # Ensure grayscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First-time initialization
        if self.prev_gray is None:
            self.prev_gray = gray
            self.reset_features(gray)
            return self.position.copy(), np.zeros(2)

        velocity = np.zeros(2)
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
        return self.position.copy(), velocity


def main():
    cap = cv2.VideoCapture(0)
    vo = VisualOdometry(motion_scale=0.2)

    # Optional: tailor capture properties
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    start_time = time.time()
    prev_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            position, velocity = vo.process(frame)
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            if current_time - start_time < 3:
                vo.position = (0, 0)

            # Output
            print(f"Position: {position}, Velocity: {velocity}, dt: {dt:.4f}s")

            # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


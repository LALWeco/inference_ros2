import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class MotionEstimate:
    """Container for motion estimation results."""
    translation_x: float
    translation_y: float
    uncertainty_x: float
    uncertainty_y: float
    success: bool = True

class MotionEstimator:
    """Estimates frame-to-frame motion using feature detection and matching."""
    
    def __init__(
        self,
        max_size: int = 512,
        max_features: int = 500,
        min_matches: int = 10,
        ransac_threshold: float = 8.0,
        max_ransac_iterations: int = 200
    ):
        """Initialize motion estimator.
        
        Args:
            max_size: Maximum image dimension for processing
            max_features: Maximum number of features to detect
            min_matches: Minimum number of matches required
            ransac_threshold: RANSAC threshold for homography estimation
            max_ransac_iterations: Maximum RANSAC iterations
        """
        self.max_size = max_size
        self.max_features = max_features
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.max_ransac_iterations = max_ransac_iterations
        
        # Initialize feature detector and descriptor
        self.detector = cv2.FastFeatureDetector_create(threshold=20)
        self.descriptor = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for motion estimation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of preprocessed image and scale factor
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Scale image if needed
        scale_factor = max(gray.shape[0] / self.max_size, 
                         gray.shape[1] / self.max_size)
        
        if scale_factor > 1:
            new_size = (
                int(gray.shape[1] / scale_factor),
                int(gray.shape[0] / scale_factor)
            )
            gray = cv2.resize(gray, new_size)
            
        return gray, scale_factor
        
    def estimate_motion(
        self,
        curr_frame: np.ndarray,
        prev_frame: Optional[np.ndarray]
    ) -> MotionEstimate:
        """Estimate motion between consecutive frames.
        
        Args:
            curr_frame: Current frame
            prev_frame: Previous frame
            
        Returns:
            MotionEstimate object containing results
        """
        # Handle first frame
        if prev_frame is None:
            return MotionEstimate(0.0, -40.0, 10.0, 10.0)
            
        # Preprocess frames
        curr_gray, scale = self._preprocess_image(curr_frame)
        prev_gray, _ = self._preprocess_image(prev_frame)
        
        # Detect features
        prev_kp = self.detector.detect(prev_gray, None)
        curr_kp = self.detector.detect(curr_gray, None)
        
        if len(prev_kp) < self.min_matches or len(curr_kp) < self.min_matches:
            return MotionEstimate(0.0, -40.0, 10.0, 10.0, success=False)
        
        # Compute descriptors
        prev_kp, prev_des = self.descriptor.compute(prev_gray, prev_kp)
        curr_kp, curr_des = self.descriptor.compute(curr_gray, curr_kp)
        
        if prev_des is None or curr_des is None:
            return MotionEstimate(0.0, -40.0, 10.0, 10.0, success=False)
        
        # Match features
        matches = self.matcher.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        if len(matches) < self.min_matches:
            return MotionEstimate(0.0, -40.0, 10.0, 10.0, success=False)
        
        # Extract matched keypoints
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography
        H, mask = cv2.findHomography(
            prev_pts,
            curr_pts,
            cv2.RANSAC,
            self.ransac_threshold,
            maxIters=self.max_ransac_iterations
        )
        
        if H is None:
            return MotionEstimate(0.0, -40.0, 10.0, 10.0, success=False)
        
        # Extract translation from homography
        tx = H[0, 2] * scale
        ty = H[1, 2] * scale
        
        # Estimate uncertainty based on number of matches
        uncertainty = (20.0 / len(matches), 20.0 / len(matches))
        
        return MotionEstimate(tx, ty, uncertainty[0], uncertainty[1])

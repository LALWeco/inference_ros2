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
    """Estimates frame-to-frame motion using feature detection and matching.
    Optimized for downward-facing camera tracking ground motion."""
    
    def __init__(
        self,
        max_size: int = 320,
        max_features: int = 500,
        min_matches: int = 10,
        ransac_threshold: float = 8.0,
        max_ransac_iterations: int = 200,
        use_cuda: bool = True
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
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # Use Shi-Tomasi corner detector - better for planar surfaces
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.1,
            minDistance=10,
            blockSize=5
        )
        
        # For optical flow - optimized for ground tracking
        self.lk_params = dict(
            winSize=(11, 11),  # Smaller window for speed
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            minEigThreshold=0.001  # Helps with floor textures
        )
        
        # State variables for optical flow tracking
        self.prev_gray = None
        self.prev_pts = None
        self.grid_size = 8  # Reduced from 16
        
        # Pre-allocate arrays to avoid repeated memory allocation
        self._temp_points = np.zeros((self.max_features, 2), dtype=np.float32)
        self._good_mask = np.zeros(self.max_features, dtype=bool)
        
        if self.use_cuda:
            # Initialize CUDA streams for async operations
            self.cuda_stream = cv2.cuda.Stream()
            # Pre-allocate GPU memory
            self.gpu_prev_gray = None
            self.gpu_curr_gray = None
            
            # CUDA optical flow detector
            self.cuda_optical_flow = cv2.cuda.SparsePyrLKOpticalFlow_create(
                winSize=(11, 11),
                maxLevel=3,
                iters=10
            )
        
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
            
        # Scale image if needed - using INTER_AREA for better quality downsampling
        scale_factor = max(gray.shape[0] / self.max_size, 
                         gray.shape[1] / self.max_size)
        
        if scale_factor > 1:
            new_size = (
                int(gray.shape[1] / scale_factor),
                int(gray.shape[0] / scale_factor)
            )
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
            
        return gray, scale_factor
    
    def _detect_grid_features(self, image: np.ndarray) -> np.ndarray:
        """Detect features in a grid pattern for more uniform distribution.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Array of feature points
        """
        h, w = image.shape
        grid_h, grid_w = h // self.grid_size, w // self.grid_size
        
        # Initialize empty point array
        all_points = []
        
        # Iterate through grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Define cell boundaries
                x1 = j * grid_w
                y1 = i * grid_h
                x2 = min((j + 1) * grid_w, w)
                y2 = min((i + 1) * grid_h, h)
                
                # Skip cells that are too small
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                
                # Extract cell and find corners
                cell = image[y1:y2, x1:x2]
                corners = cv2.goodFeaturesToTrack(
                    cell, 
                    maxCorners=2,  # Only get the strongest 2 corners per cell
                    qualityLevel=0.1, 
                    minDistance=5
                )
                
                if corners is not None:
                    # Adjust coordinates to full image
                    corners[:, 0, 0] += x1
                    corners[:, 0, 1] += y1
                    all_points.extend(corners.reshape(-1, 2))
        
        return np.array(all_points, dtype=np.float32) if all_points else np.array([], dtype=np.float32)
    
    def _detect_grid_features_optimized(self, image: np.ndarray) -> np.ndarray:
        """Optimized grid feature detection with reduced allocations."""
        h, w = image.shape
        grid_h, grid_w = h // self.grid_size, w // self.grid_size
        
        points_count = 0
        max_points_per_cell = 1  # Reduced from 2
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if points_count >= self.max_features - max_points_per_cell:
                    break
                    
                x1, y1 = j * grid_w, i * grid_h
                x2, y2 = min((j + 1) * grid_w, w), min((i + 1) * grid_h, h)
                
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                
                cell = image[y1:y2, x1:x2]
                corners = cv2.goodFeaturesToTrack(
                    cell, 
                    maxCorners=max_points_per_cell,
                    qualityLevel=0.15,  # Slightly higher threshold
                    minDistance=8  # Slightly larger distance
                )
                
                if corners is not None:
                    for corner in corners:
                        if points_count < self.max_features:
                            self._temp_points[points_count, 0] = corner[0, 0] + x1
                            self._temp_points[points_count, 1] = corner[0, 1] + y1
                            points_count += 1
        
        return self._temp_points[:points_count].copy() if points_count > 0 else np.array([], dtype=np.float32)
        
    def _preprocess_image_cuda(self, image: np.ndarray) -> Tuple[cv2.cuda.GpuMat, float]:
        """CUDA-accelerated image preprocessing."""
        # Upload to GPU
        gpu_image = cv2.cuda.GpuMat()
        gpu_image.upload(image)
        
        # Convert to grayscale on GPU
        if len(image.shape) == 3:
            gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
        else:
            gpu_gray = gpu_image
            
        # Scale image if needed
        scale_factor = max(image.shape[0] / self.max_size, 
                          image.shape[1] / self.max_size)
        
        if scale_factor > 1:
            new_size = (
                int(image.shape[1] / scale_factor),
                int(image.shape[0] / scale_factor)
            )
            gpu_resized = cv2.cuda.resize(gpu_gray, new_size, interpolation=cv2.INTER_AREA)
            return gpu_resized, scale_factor
            
        return gpu_gray, scale_factor
    
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
            # Initialize state for optical flow tracking
            self.prev_gray, _ = self._preprocess_image(curr_frame)
            
            # Detect initial features using grid approach for better distribution
            self.prev_pts = self._detect_grid_features(self.prev_gray)
            
            if len(self.prev_pts) < self.min_matches:
                # Fallback to standard corner detection if grid approach fails
                corners = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
                if corners is not None:
                    self.prev_pts = corners.reshape(-1, 2)
                else:
                    self.prev_pts = np.array([], dtype=np.float32)
            
            # Return default zero motion for first frame
            return MotionEstimate(0.0, 0.0, 10.0, 10.0)
            
        # Preprocess current frame
        curr_gray, scale = self._preprocess_image(curr_frame)
        
        # If we don't have previous state, initialize from prev_frame
        if self.prev_gray is None:
            self.prev_gray, _ = self._preprocess_image(prev_frame)
            # Detect features in previous frame
            self.prev_pts = self._detect_grid_features(self.prev_gray)
            
            if len(self.prev_pts) < self.min_matches:
                # Try standard feature detection if grid approach fails
                corners = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
                if corners is not None:
                    self.prev_pts = corners.reshape(-1, 2)
                else:
                    # Not enough features to track
                    self.prev_gray = curr_gray
                    return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Make sure we have points to track
        if self.prev_pts is None or len(self.prev_pts) < self.min_matches:
            # Reinitialize feature detection
            self.prev_pts = self._detect_grid_features(self.prev_gray)
            
            if len(self.prev_pts) < self.min_matches:
                # Fallback to standard corner detection
                corners = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
                if corners is None or len(corners) < self.min_matches:
                    # Still not enough features
                    self.prev_gray = curr_gray
                    return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
                self.prev_pts = corners.reshape(-1, 2)
        
        # Track features using optical flow (much faster than re-detecting)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, 
            self.prev_pts.reshape(-1, 1, 2), None, 
            **self.lk_params
        )
        
        # Check if tracking was successful
        if curr_pts is None:
            self.prev_gray = curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Keep only good points
        good_mask = (status == 1) & (err < 10)  # Only keep low-error points
        good_old = self.prev_pts[good_mask.ravel()]
        good_new = curr_pts.reshape(-1, 2)[good_mask.ravel()]
        
        # Check if we have enough good matches
        if len(good_new) < self.min_matches:
            # Reset for next frame
            self.prev_gray = curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Use a simpler transformation model for 2D ground plane motion
        # estimateAffinePartial2D is much faster than full homography for planar motion
        transform, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=self.max_ransac_iterations,
            confidence=0.99
        )
        
        if transform is None or inliers is None:
            self.prev_gray = curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Extract translation
        tx = transform[0, 2] * scale
        ty = transform[1, 2] * scale
        
        # Count inliers for uncertainty estimation (lower means higher uncertainty)
        inlier_count = np.sum(inliers)
        uncertainty = max(20.0 / (inlier_count + 1), 1.0)
        
        # Update points for next iteration - use only inlier points
        if inlier_count > 0:
            inlier_mask = inliers.ravel() == 1
            self.prev_pts = good_new[inlier_mask]
        else:
            self.prev_pts = good_new
        
        # Periodically refresh points to avoid drift (every ~5 frames)
        if np.random.random() < 0.2 or len(self.prev_pts) < self.min_matches * 1.5:
            # Add new points to existing ones
            new_pts = self._detect_grid_features(curr_gray)
            if len(new_pts) > 0:
                self.prev_pts = np.vstack([self.prev_pts, new_pts]) if len(self.prev_pts) > 0 else new_pts
        
        # Update previous frame
        self.prev_gray = curr_gray
        
        # Return motion estimate with same format as original
        return MotionEstimate(tx, ty, uncertainty, uncertainty, success=True)
    
    def reset(self):
        """Reset the estimator state."""
        self.prev_gray = None
        self.prev_pts = None
    
    def _detect_features_cuda(self, gpu_image: cv2.cuda.GpuMat) -> np.ndarray:
        """CUDA-accelerated feature detection."""
        # Use CUDA corner detector
        cuda_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
            srcType=0,
            maxCorners=self.max_features,
            qualityLevel=0.1,
            minDistance=10,
            blockSize=5
        )
        
        gpu_corners = cuda_detector.detect(gpu_image)
        
        if gpu_corners.empty():
            return np.array([], dtype=np.float32)
        
        # Download corners
        corners = gpu_corners.download()
        return corners.reshape(-1, 2)
    
    def estimate_motion_cuda(
        self,
        curr_frame: np.ndarray,
        prev_frame: Optional[np.ndarray]
    ) -> MotionEstimate:
        """CUDA-accelerated motion estimation."""
        
        # Handle first frame
        if prev_frame is None:
            self.gpu_prev_gray, _ = self._preprocess_image_cuda(curr_frame)
            # Download to CPU only for feature detection (could be optimized further)
            # cpu_gray = self.gpu_prev_gray.download()
            self.prev_pts = self._detect_features_cuda(self.gpu_prev_gray)
            return MotionEstimate(0.0, 0.0, 10.0, 10.0)
        
        # Preprocess current frame on GPU
        gpu_curr_gray, scale = self._preprocess_image_cuda(curr_frame)
        
        if self.gpu_prev_gray is None:
            self.gpu_prev_gray, _ = self._preprocess_image_cuda(prev_frame)
            # cpu_prev = self.gpu_prev_gray.download()
            self.prev_pts = self._detect_features_cuda(self.gpu_prev_gray)
            
            if len(self.prev_pts) < self.min_matches:
                self.gpu_prev_gray = gpu_curr_gray
                return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        if self.prev_pts is None or len(self.prev_pts) < self.min_matches:
            # cpu_prev = self.gpu_prev_gray.download()
            self.prev_pts = self._detect_features_cuda(self.gpu_prev_gray)
            
            if len(self.prev_pts) < self.min_matches:
                self.gpu_prev_gray = gpu_curr_gray
                return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Upload points to GPU
        gpu_prev_pts = cv2.cuda.GpuMat()
        # Ensure points are in the correct format for CUDA 1 x N x 2 and 32FC2 
        gpu_prev_pts.upload(self.prev_pts.reshape(1, -1, 2).astype(np.float32))
        
        # CUDA optical flow - this is where the major speedup happens
        gpu_curr_pts, gpu_status, gpu_err = self.cuda_optical_flow.calc(
            self.gpu_prev_gray, gpu_curr_gray, gpu_prev_pts, None
        )
        
        # Download results
        curr_pts = gpu_curr_pts.download()
        status = gpu_status.download()
        err = gpu_err.download()
        
        # Check if tracking was successful
        if curr_pts is None:
            self.prev_gray = gpu_curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Keep only good points
        good_mask = (status == 1) & (err < 10)  # Only keep low-error points
        good_old = self.prev_pts[good_mask.ravel()]
        good_new = curr_pts.reshape(-1, 2)[good_mask.ravel()]
        
        # Check if we have enough good matches
        if len(good_new) < self.min_matches:
            # Reset for next frame
            self.prev_gray = gpu_curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Use a simpler transformation model for 2D ground plane motion
        # estimateAffinePartial2D is much faster than full homography for planar motion
        transform, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=self.max_ransac_iterations,
            confidence=0.99
        )
        
        if transform is None or inliers is None:
            self.prev_gray = gpu_curr_gray
            return MotionEstimate(0.0, 0.0, 10.0, 10.0, success=False)
        
        # Extract translation
        tx = transform[0, 2] * scale
        ty = transform[1, 2] * scale
        
        # Count inliers for uncertainty estimation (lower means higher uncertainty)
        inlier_count = np.sum(inliers)
        uncertainty = max(20.0 / (inlier_count + 1), 1.0)
        
        # Update points for next iteration - use only inlier points
        if inlier_count > 0:
            inlier_mask = inliers.ravel() == 1
            self.prev_pts = good_new[inlier_mask]
        else:
            self.prev_pts = good_new
        
        # Periodically refresh points to avoid drift (every ~5 frames)
        if np.random.random() < 0.2 or len(self.prev_pts) < self.min_matches * 1.5:
            # Add new points to existing ones
            new_pts = self._detect_grid_features(gpu_curr_gray)
            if len(new_pts) > 0:
                self.prev_pts = np.vstack([self.prev_pts, new_pts]) if len(self.prev_pts) > 0 else new_pts
        
        # Update previous frame
        self.prev_gray = gpu_curr_gray
        
        # Return motion estimate with same format as original
        return MotionEstimate(tx, ty, uncertainty, uncertainty, success=True)
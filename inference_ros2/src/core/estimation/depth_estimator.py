import numpy as np
from typing import Optional, Tuple
from geometry_msgs.msg import Point
from .base_estimator import BasePointEstimator

class DepthEstimator(BasePointEstimator):
    """3D point estimation using depth information."""
    
    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        depth_sample_size: int = 5
    ):
        """Initialize depth-based point estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            depth_sample_size: Size of depth sampling window
        """
        super().__init__(camera_matrix)
        self.depth_sample_size = depth_sample_size
        self.depth_image = None
        
    def set_depth_image(self, depth_image: np.ndarray) -> None:
        """Set current depth image.
        
        Args:
            depth_image: Depth image as numpy array
        """
        self.depth_image = depth_image
        
    def get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        """Get depth value at image coordinates with sampling window.
        
        Args:
            x: x-coordinate in image
            y: y-coordinate in image
            
        Returns:
            Median depth value or None if invalid
        """
        if self.depth_image is None:
            return None
            
        h, w = self.depth_image.shape
        size = self.depth_sample_size
        
        # Define sampling box
        x1 = max(0, x - size // 2)
        x2 = min(w, x + size // 2 + 1)
        y1 = max(0, y - size // 2)
        y2 = min(h, y + size // 2 + 1)
        
        # Get median of valid depths in window
        depth_region = self.depth_image[y1:y2, x1:x2]
        valid_depths = depth_region[depth_region > 0]
        
        return np.median(valid_depths) if len(valid_depths) > 0 else None
        
    def estimate_3d_point(self, keypoint_2d: Point) -> Optional[Point]:
        """Estimate 3D point using depth information.
        
        Args:
            keypoint_2d: 2D keypoint position
            
        Returns:
            3D point or None if estimation fails
        """
        if self.depth_image is None:
            return None
            
        # Get depth at keypoint
        depth = self.get_depth_at_point(
            int(keypoint_2d.x),
            int(keypoint_2d.y)
        )
        
        if depth is None:
            return None
            
        # Back-project to 3D using normalized coordinates
        ray = self._normalize_image_point(keypoint_2d.x, keypoint_2d.y)
        
        point_3d = Point()
        point_3d.z = float(depth)
        point_3d.x = ray[0] * depth
        point_3d.y = ray[1] * depth
        
        return point_3d

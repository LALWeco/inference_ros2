from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from geometry_msgs.msg import Point

class BasePointEstimator(ABC):
    """Base class for 3D point estimation methods."""
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        """Initialize the point estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
    
    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        """Set camera intrinsic matrix.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
    
    @abstractmethod
    def estimate_3d_point(self, keypoint_2d: Point) -> Optional[Point]:
        """Estimate 3D point from 2D keypoint.
        
        Args:
            keypoint_2d: 2D keypoint position
            
        Returns:
            3D point or None if estimation fails
        """
        pass
    
    def _normalize_image_point(self, x: float, y: float) -> np.ndarray:
        """Convert image coordinates to normalized camera coordinates.
        
        Args:
            x: x-coordinate in image
            y: y-coordinate in image
            
        Returns:
            Normalized point coordinates as numpy array
        """
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
            
        x_norm = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_norm = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        return np.array([x_norm, y_norm, 1.0])

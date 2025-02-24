import numpy as np
from typing import Optional
from geometry_msgs.msg import Point
from .base_estimator import BasePointEstimator

class GeometricEstimator(BasePointEstimator):
    """3D point estimation using geometric method with camera height and tilt."""
    
    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        camera_height: float = 1.0,
        camera_tilt_degrees: float = 30.0
    ):
        """Initialize geometric point estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            camera_height: Height of camera from ground (meters)
            camera_tilt_degrees: Camera tilt angle in degrees (0° is horizontal, positive tilts down)
        """
        super().__init__(camera_matrix)
        self.camera_height = camera_height
        self.set_camera_tilt(camera_tilt_degrees)
        
    def set_camera_height(self, height: float) -> None:
        """Set camera height from ground.
        
        Args:
            height: Camera height in meters
        """
        self.camera_height = height
        
    def set_camera_tilt(self, tilt_degrees: float) -> None:
        """Set camera tilt angle.
        
        Args:
            tilt_degrees: Tilt angle in degrees
        """
        # Convert to radians, 90° - tilt to make 0° look down
        self.camera_tilt = np.radians(90.0 - tilt_degrees)
        
    def estimate_3d_point(self, keypoint_2d: Point) -> Optional[Point]:
        """Estimate 3D point using geometric method.
        
        This method assumes the point lies on the ground plane and uses
        the camera height and tilt angle to compute its 3D position.
        
        Args:
            keypoint_2d: 2D keypoint position
            
        Returns:
            3D point or None if estimation fails
        """
        # Get normalized ray direction
        ray = self._normalize_image_point(keypoint_2d.x, keypoint_2d.y)
        ray = ray / np.linalg.norm(ray)
        
        # Apply camera tilt rotation to ray
        ray_rotated = np.array([
            ray[0],
            ray[1] * np.cos(self.camera_tilt) - ray[2] * np.sin(self.camera_tilt),
            ray[1] * np.sin(self.camera_tilt) + ray[2] * np.cos(self.camera_tilt)
        ])
        
        # Check if ray intersects ground plane
        if ray_rotated[2] == 0:  # Parallel to ground
            return None
            
        # Find intersection with ground plane (z = 0)
        t = -self.camera_height / ray_rotated[2]
        
        # Create 3D point at intersection
        point_3d = Point()
        point_3d.x = t * ray_rotated[0]
        point_3d.y = t * ray_rotated[1]
        point_3d.z = 0.0  # On ground plane
        
        return point_3d

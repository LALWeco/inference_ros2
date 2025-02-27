import numpy as np
from typing import Optional
from geometry_msgs.msg import Point
from .base_estimator import BasePointEstimator

class GeometricEstimator(BasePointEstimator):
    """3D point estimation using geometric method with known camera height."""
    
    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        camera_height: float = 1.0,
        camera_tilt_degrees: float = 90.0  # Kept for compatibility, but fixed at 90°
    ):
        """Initialize geometric point estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            camera_height: Height of camera from ground (meters)
            camera_tilt_degrees: Ignored, camera is assumed to be looking straight down
        """
        super().__init__(camera_matrix)
        self.camera_height = camera_height
        
        # For a downward-facing camera:
        # - Camera's Z (forward) becomes world's -Z (up)
        # - Camera's Y (down in image) becomes world's -Y (forward)  
        # - Camera's X (right) stays as world's X (right)
        self.cam_to_world = np.array([
            [1.0, 0.0, 0.0],   # Camera X → World X
            [0.0, -1.0, 0.0],  # Camera Y → World -Y 
            [0.0, 0.0, -1.0]   # Camera Z → World -Z
        ])
        
    def set_camera_height(self, height: float) -> None:
        """Set camera height from ground.
        
        Args:
            height: Camera height in meters
        """
        if height <= 0:
            raise ValueError("Camera height must be positive")
        self.camera_height = height
        
    def estimate_3d_point(self, keypoint_2d: Point) -> Optional[Point]:
        """Estimate 3D point location using geometric method.
        
        The process:
        1. Convert image point to ray in camera coordinates (standard pinhole model)
        2. Transform ray to world coordinates (camera looking down)
        3. Find intersection with ground plane (Z = -camera_height)
        
        Args:
            keypoint_2d: 2D keypoint position in image coordinates (pixels)
            
        Returns:
            3D point in world coordinates or None if estimation fails
        """
        if self.camera_matrix is None:
            print("Camera matrix not set")
            return None
            
        print(f"Processing image point: ({keypoint_2d.x}, {keypoint_2d.y})")
            
        # Get normalized ray in camera coordinates
        cam_ray = self._normalize_image_point(keypoint_2d.x, keypoint_2d.y)
        if cam_ray is None:
            print("Failed to normalize image point")
            return None
            
        print(f"Camera-space ray: {cam_ray}")
            
        # Transform to world coordinates
        world_ray = self.cam_to_world @ cam_ray
        print(f"World-space ray: {world_ray}")
            
        # In world coordinates:
        # - Camera is at origin looking down -Z
        # - Ground plane is at z = -camera_height
        # - We want world_ray.z negative to hit ground
        if world_ray[2] >= 0:
            print(f"Invalid ray direction - not pointing down (z={world_ray[2]})")
            return None
            
        # Calculate scaling factor to reach ground plane
        # At intersection: ray.z * scale = -camera_height
        scale = -self.camera_height / world_ray[2]
        print(f"Scale factor: {scale}")
        
        # Calculate intersection point
        intersection = scale * world_ray
        print(f"Intersection point: {intersection}")
        
        # Return world-space point (convert to ROS convention where Z is up)
        point_3d = Point()
        point_3d.x = float(intersection[0])    # Right
        point_3d.y = float(intersection[1])    # Forward
        point_3d.z = float(-intersection[2])   # Up (negate Z to match ROS convention)
        
        return point_3d
        
    def _normalize_image_point(self, x: float, y: float) -> Optional[np.ndarray]:
        """Convert image coordinates to normalized ray direction in camera space.
        
        Args:
            x: X coordinate in image (pixels)
            y: Y coordinate in image (pixels)
            
        Returns:
            3D ray direction vector in camera coordinates or None if conversion fails
        """
        try:
            # Convert to homogeneous coordinates
            point_2d = np.array([x, y, 1.0])
            
            # Get normalized ray by applying inverse camera matrix
            # This gives us direction in standard camera coordinates
            # where Z points forward along optical axis
            ray = np.linalg.solve(self.camera_matrix, point_2d)
            
            # Normalize to unit vector
            ray = ray / np.linalg.norm(ray)
            
            return ray
            
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Error normalizing image point: {e}")
            return None

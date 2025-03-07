import cv2
import numpy as np
from typing import List, Tuple, Optional
from bytetracker.byte_tracker import BYTETracker, STrack

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    class_names: List[str],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw detection boxes and keypoints on image.
    
    Args:
        image: Input image to draw on
        boxes: Array of bounding boxes (N, 4) in xyxy format
        keypoints: Array of keypoints (N, K, 3) where K is number of keypoints
        class_names: List of class names
        color: BGR color tuple for drawing
        
    Returns:
        Image with drawings
    """
    img = image.copy()
    
    if boxes.shape[0] != 0:
        for i in range(boxes.shape[0]):
            box = boxes[i]
            kpts = keypoints[i]
            
            # Draw bounding box
            x1, y1, x2, y2 = box[:4].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            class_id = int(box[5]) if len(box) > 5 else 0
            class_name = class_names[class_id]
            conf = round(box[4], 2) if len(box) > 4 else 1.0
            label = f"{class_name} {conf:.2f}"
            
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw keypoints
            for kpt in kpts:
                x, y, conf = kpt
                if conf > 0:
                    cv2.circle(img, (int(x), int(y)), 4, color, -1)
    
    return img

def draw_tracks(
    image: np.ndarray,
    tracks: List[STrack],
    draw_history: bool = True,
    active_color: Tuple[int, int, int] = (0, 255, 0),
    lost_color: Tuple[int, int, int] = (128, 128, 128),
    history_color: Tuple[int, int, int] = (255, 0, 255)
) -> np.ndarray:
    """Draw tracking results on image.
    
    Args:
        image: Input image to draw on
        tracks: List of STrack objects
        draw_history: Whether to draw track history
        active_color: BGR color for active tracks
        lost_color: BGR color for lost tracks
        history_color: BGR color for track history
        
    Returns:
        Image with drawings
    """
    img = image.copy()
    
    # Draw active tracks
    for track in tracks:
        if track.is_activated:
            # Get track info
            tlbr = track._detection.astype(int)
            tid = int(track.track_id)
            
            # Draw bounding box
            cv2.rectangle(img, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), active_color, 2)
            
            # Draw track ID
            cv2.putText(
                img,
                f"ID: {tid}",
                (tlbr[0], tlbr[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                active_color,
                2
            )
            
            # Draw Kalman point
            if hasattr(track, 'mean'):
                kx, ky = track.mean[0:2].astype(int)
                cv2.circle(img, (kx, ky), 4, history_color, -1)
            
            # Draw track history
            if draw_history and len(track.track_history) > 1:
                # Convert history points to integer array
                points = np.array(track.track_history, dtype=np.int32)
                
                # Draw history line
                cv2.polylines(img, [points], False, history_color, 2)
                
                # Draw history points
                for point in points:
                    cv2.circle(img, tuple(point), 3, active_color, -1)
    
    # Draw lost tracks with faded appearance
    if draw_history:
        overlay = img.copy()
        for track in [t for t in tracks if not t.is_activated]:
            if len(track.track_history) > 1:
                points = np.array(track.track_history, dtype=np.int32)
                cv2.polylines(overlay, [points], False, lost_color, 1)
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    return img

def draw_points_with_uncertainty(
    image: np.ndarray,
    points: np.ndarray,
    uncertainties: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 4
) -> np.ndarray:
    """Draw points with uncertainty ellipses.
    
    Args:
        image: Input image to draw on
        points: Array of points (N, 2)
        uncertainties: Array of uncertainties (N, 2)
        color: BGR color tuple
        radius: Point radius
        
    Returns:
        Image with drawings
    """
    img = image.copy()
    
    for i in range(len(points)):
        pt = tuple(points[i].astype(int))
        uncertainty = uncertainties[i]
        
        # Draw center point
        cv2.circle(img, pt, radius, color, -1)
        
        # Draw uncertainty ellipse
        axes = tuple((3 * uncertainty).astype(int))  # 3-sigma ellipse
        cv2.ellipse(img, pt, axes, 0, 0, 360, color, 1)
    
    return img

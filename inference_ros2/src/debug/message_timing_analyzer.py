#!/usr/bin/env python3
"""
Debug script to analyze ROS2 message timing and synchronization issues.
Run this alongside your motion tracking node to diagnose problems.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from lalweco_perception_msgs.msg import Keypoint2DArray
import time
from collections import deque
import statistics

class MessageTimingAnalyzer(Node):
    """Analyze message timing patterns to debug synchronization issues."""
    
    def __init__(self):
        super().__init__('message_timing_analyzer')
        
        # Timing storage
        self.image_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.image_intervals = deque(maxlen=100)
        self.detection_intervals = deque(maxlen=100)
        self.image_ages = deque(maxlen=100)
        self.detection_ages = deque(maxlen=100)
        
        # Last message times
        self.last_image_time = None
        self.last_detection_time = None
        
        # Counters
        self.image_count = 0
        self.detection_count = 0
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/oak/rgb/image_raw/compressed',
            self.image_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            Keypoint2DArray,
            '/inference/Keypoint2DDetArray',
            self.detection_callback,
            10
        )
        
        # Analysis timer
        self.create_timer(10.0, self.analyze_and_report)  # Every 10 seconds
        
        self.get_logger().info("Message Timing Analyzer started")
    
    def image_callback(self, msg):
        """Analyze image message timing."""
        current_time = time.time()
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = (current_time - msg_time) * 1000  # Age in ms
        
        self.image_count += 1
        self.image_times.append(current_time)
        self.image_ages.append(age)
        
        if self.last_image_time is not None:
            interval = (current_time - self.last_image_time) * 1000  # Interval in ms
            self.image_intervals.append(interval)
            
        self.last_image_time = current_time
    
    def detection_callback(self, msg):
        """Analyze detection message timing."""
        current_time = time.time()
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = (current_time - msg_time) * 1000  # Age in ms
        
        self.detection_count += 1
        self.detection_times.append(current_time)
        self.detection_ages.append(age)
        
        if self.last_detection_time is not None:
            interval = (current_time - self.last_detection_time) * 1000  # Interval in ms
            self.detection_intervals.append(interval)
            
        self.last_detection_time = current_time
    
    def analyze_and_report(self):
        """Analyze collected timing data and report statistics."""
        if len(self.image_intervals) < 5 or len(self.detection_intervals) < 5:
            self.get_logger().warn("Not enough data for analysis yet...")
            return
        
        # Calculate statistics
        img_stats = self.calculate_stats(self.image_intervals, "Image intervals")
        det_stats = self.calculate_stats(self.detection_intervals, "Detection intervals")
        img_age_stats = self.calculate_stats(self.image_ages, "Image ages")
        det_age_stats = self.calculate_stats(self.detection_ages, "Detection ages")
        
        # Calculate expected vs actual rates
        if len(self.image_times) >= 2:
            total_time = self.image_times[-1] - self.image_times[0]
            img_rate = (len(self.image_times) - 1) / total_time
        else:
            img_rate = 0
            
        if len(self.detection_times) >= 2:
            total_time = self.detection_times[-1] - self.detection_times[0]
            det_rate = (len(self.detection_times) - 1) / total_time
        else:
            det_rate = 0
        
        # Report findings
        self.get_logger().warn("=== TIMING ANALYSIS REPORT ===")
        self.get_logger().warn(f"Image messages: {self.image_count} total, {img_rate:.1f} Hz")
        self.get_logger().warn(f"Detection messages: {self.detection_count} total, {det_rate:.1f} Hz")
        
        self.get_logger().warn(f"Image intervals (ms): avg={img_stats['mean']:.1f}, "
                              f"std={img_stats['std']:.1f}, min={img_stats['min']:.1f}, "
                              f"max={img_stats['max']:.1f}")
        
        self.get_logger().warn(f"Detection intervals (ms): avg={det_stats['mean']:.1f}, "
                              f"std={det_stats['std']:.1f}, min={det_stats['min']:.1f}, "
                              f"max={det_stats['max']:.1f}")
        
        self.get_logger().warn(f"Image ages (ms): avg={img_age_stats['mean']:.1f}, "
                              f"std={img_age_stats['std']:.1f}, min={img_age_stats['min']:.1f}, "
                              f"max={img_age_stats['max']:.1f}")
        
        self.get_logger().warn(f"Detection ages (ms): avg={det_age_stats['mean']:.1f}, "
                              f"std={det_age_stats['std']:.1f}, min={det_age_stats['min']:.1f}, "
                              f"max={det_age_stats['max']:.1f}")
        
        # Diagnose issues
        self.diagnose_issues(img_stats, det_stats, img_age_stats, det_age_stats, img_rate, det_rate)
        
        self.get_logger().warn("=== END REPORT ===")
    
    def calculate_stats(self, data, label):
        """Calculate statistics for a data series."""
        if not data:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': statistics.mean(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data)
        }
    
    def diagnose_issues(self, img_stats, det_stats, img_age_stats, det_age_stats, img_rate, det_rate):
        """Diagnose potential issues based on statistics."""
        issues = []
        
        # Check for irregular intervals
        if img_stats['std'] > 50:  # High variance in image intervals
            issues.append(f"Irregular image timing (std={img_stats['std']:.1f}ms)")
        
        if det_stats['std'] > 50:  # High variance in detection intervals
            issues.append(f"Irregular detection timing (std={det_stats['std']:.1f}ms)")
        
        # Check for high message ages (clock issues)
        if img_age_stats['mean'] > 1000:  # Average age > 1 second
            issues.append(f"High image message age (avg={img_age_stats['mean']:.1f}ms) - clock sync issue!")
        
        if det_age_stats['mean'] > 1000:  # Average age > 1 second
            issues.append(f"High detection message age (avg={det_age_stats['mean']:.1f}ms) - clock sync issue!")
        
        # Check for low rates
        if img_rate < 5:  # Less than 5 Hz
            issues.append(f"Low image rate ({img_rate:.1f} Hz)")
        
        if det_rate < 5:  # Less than 5 Hz
            issues.append(f"Low detection rate ({det_rate:.1f} Hz)")
        
        # Check for rate mismatch
        if abs(img_rate - det_rate) > 2:  # More than 2 Hz difference
            issues.append(f"Rate mismatch: images={img_rate:.1f}Hz, detections={det_rate:.1f}Hz")
        
        # Report issues
        if issues:
            self.get_logger().error("ISSUES DETECTED:")
            for issue in issues:
                self.get_logger().error(f"  - {issue}")
        else:
            self.get_logger().info("No major timing issues detected")
        
        # Recommendations
        recommendations = []
        
        if img_age_stats['mean'] > 1000 or det_age_stats['mean'] > 1000:
            recommendations.append("Fix clock synchronization between nodes/machines")
            recommendations.append("Check use_sim_time parameter if using bag files")
        
        if img_stats['std'] > 100 or det_stats['std'] > 100:
            recommendations.append("Increase synchronizer tolerance (sync_tolerance parameter)")
            recommendations.append("Enable fallback mode for unreliable timing")
        
        if img_rate < 10 or det_rate < 10:
            recommendations.append("Check network latency and bandwidth")
            recommendations.append("Increase queue sizes for message buffering")
        
        if recommendations:
            self.get_logger().info("RECOMMENDATIONS:")
            for rec in recommendations:
                self.get_logger().info(f"  - {rec}")

def main():
    rclpy.init()
    analyzer = MessageTimingAnalyzer()
    
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

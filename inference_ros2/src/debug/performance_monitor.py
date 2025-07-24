#!/usr/bin/env python3
"""
Real-time performance monitor for motion tracking node.
Run this in a separate terminal to monitor performance trends.
"""

import rclpy
from rclpy.node import Node
from lalweco_perception_msgs.msg import Keypoint2DArray
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

class PerformanceMonitor(Node):
    """Monitor the performance of the motion tracking node in real-time."""
    
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Performance tracking
        self.message_times = deque(maxlen=100)
        self.message_counts = deque(maxlen=100)
        self.rates = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        
        self.last_msg_time = None
        self.msg_count = 0
        self.window_start = time.time()
        
        # Subscribe to tracking output
        self.tracking_sub = self.create_subscription(
            Keypoint2DArray,
            '/tracking/tracked_keypoints',
            self.tracking_callback,
            10
        )
        
        # Timer for rate calculation
        self.create_timer(1.0, self.calculate_rate)  # Every second
        
        self.get_logger().info("Performance Monitor started - watching /tracking/tracked_keypoints")
    
    def tracking_callback(self, msg):
        """Process tracking messages and collect performance data."""
        current_time = time.time()
        self.msg_count += 1
        
        # Store message timestamp and detection count
        self.message_times.append(current_time)
        self.detection_counts.append(len(msg.detections))
        
        # Calculate message interval if we have a previous message
        if self.last_msg_time is not None:
            interval = (current_time - self.last_msg_time) * 1000  # ms
            if interval > 200:  # Log slow intervals
                self.get_logger().warn(f"Slow tracking message interval: {interval:.1f}ms")
        
        self.last_msg_time = current_time
    
    def calculate_rate(self):
        """Calculate and log the current message rate."""
        current_time = time.time()
        window_duration = current_time - self.window_start
        
        if window_duration >= 1.0:  # At least 1 second of data
            rate = self.msg_count / window_duration
            self.rates.append(rate)
            
            # Calculate average detection count
            avg_detections = sum(self.detection_counts) / len(self.detection_counts) if self.detection_counts else 0
            
            # Log performance
            self.get_logger().info(
                f"Tracking Rate: {rate:.1f} Hz, "
                f"Avg Detections: {avg_detections:.1f}, "
                f"Messages: {self.msg_count}"
            )
            
            # Warn about performance degradation
            if len(self.rates) >= 10:
                recent_avg = sum(list(self.rates)[-5:]) / 5  # Last 5 seconds average
                overall_avg = sum(self.rates) / len(self.rates)
                
                if recent_avg < overall_avg * 0.7:  # 30% drop
                    self.get_logger().warn(
                        f"Performance degradation detected! "
                        f"Recent: {recent_avg:.1f} Hz vs Overall: {overall_avg:.1f} Hz"
                    )
            
            # Reset counters
            self.msg_count = 0
            self.window_start = current_time

class PlotMonitor:
    """Real-time plotting of performance metrics."""
    
    def __init__(self, monitor_node):
        self.monitor = monitor_node
        
        # Set up the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Motion Tracking Performance Monitor')
        
        # Rate plot
        self.ax1.set_title('Tracking Rate (Hz)')
        self.ax1.set_ylabel('Hz')
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], 'b-', label='Rate')
        self.ax1.legend()
        
        # Detection count plot
        self.ax2.set_title('Detection Count per Message')
        self.ax2.set_ylabel('Count')
        self.ax2.set_xlabel('Time (samples)')
        self.ax2.grid(True)
        self.line2, = self.ax2.plot([], [], 'r-', label='Detections')
        self.ax2.legend()
        
        # Animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=1000, blit=False
        )
    
    def update_plot(self, frame):
        """Update the plots with new data."""
        # Update rate plot
        if self.monitor.rates:
            x_data = list(range(len(self.monitor.rates)))
            y_data = list(self.monitor.rates)
            self.line1.set_data(x_data, y_data)
            self.ax1.relim()
            self.ax1.autoscale_view()
        
        # Update detection count plot
        if self.monitor.detection_counts:
            x_data = list(range(len(self.monitor.detection_counts)))
            y_data = list(self.monitor.detection_counts)
            self.line2.set_data(x_data, y_data)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        return self.line1, self.line2
    
    def show(self):
        """Show the plot."""
        plt.tight_layout()
        plt.show()

def main():
    rclpy.init()
    
    # Create monitor node
    monitor = PerformanceMonitor()
    
    # Check if we should show plots
    try:
        import matplotlib
        show_plots = True
        print("Matplotlib available - will show real-time plots")
    except ImportError:
        show_plots = False
        print("Matplotlib not available - terminal logging only")
    
    if show_plots:
        # Create plot monitor in a separate thread
        plot_monitor = PlotMonitor(monitor)
        
        # Run ROS2 spinning in a separate thread
        spin_thread = threading.Thread(target=lambda: rclpy.spin(monitor))
        spin_thread.daemon = True
        spin_thread.start()
        
        # Show plots (this blocks)
        plot_monitor.show()
    else:
        # Just run the node
        try:
            rclpy.spin(monitor)
        except KeyboardInterrupt:
            pass
    
    monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

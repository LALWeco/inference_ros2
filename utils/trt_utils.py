import pycuda.driver as cuda
import tensorrt as trt
import rclpy
from rclpy.node import Node

class TrtLogger(trt.ILogger):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node

    def log(self, severity, msg):
        if severity == trt.Logger.VERBOSE:
            self.ros_node.get_logger().debug(msg)
        elif severity == trt.Logger.INFO:
            self.ros_node.get_logger().info(msg)
        elif severity == trt.Logger.WARNING:
            self.ros_node.get_logger().warn(msg)
        elif severity == trt.Logger.ERROR:
            self.ros_node.get_logger().error(msg)
        elif severity == trt.Logger.INTERNAL_ERROR:
            self.ros_node.get_logger().fatal(msg)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        # self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def _cuda_error_check(args):
    """CUDA error checking."""
    err, ret = args[0], args[1:]
    if isinstance(err, cuda.CUresult):
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
      raise RuntimeError("Unknown error type: {}".format(err))
    # Special case so that no unpacking is needed at call-site.
    if len(ret) == 1:
      return ret[0]
    return ret
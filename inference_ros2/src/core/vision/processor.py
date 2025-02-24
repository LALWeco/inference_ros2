import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class HostDeviceMem:
    """Class to hold host and device memory for TensorRT."""
    host: np.ndarray
    device: cuda.DeviceAllocation

class TensorRTProcessor:
    """Handles TensorRT model initialization and inference."""
    
    def __init__(
        self, 
        model_path: str, 
        precision: str = "fp32",
        logger: Optional[trt.Logger] = None
    ):
        """Initialize TensorRT processor.
        
        Args:
            model_path: Path to TensorRT engine file
            precision: Model precision ("fp32", "fp16", or "int8")
            logger: TensorRT logger instance
        """
        self.model_path = model_path
        self.precision = precision
        self.logger = logger or trt.Logger(trt.Logger.WARNING)
        self.dtype = self._get_dtype(precision)
        
        # CUDA initialization
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        
        # TensorRT initialization
        self._init_tensorrt()
        
    def _get_dtype(self, precision: str) -> np.dtype:
        """Get numpy dtype based on precision."""
        dtype_map = {
            "fp32": np.float32,
            "fp16": np.float16,
            "int8": np.int8
        }
        return dtype_map.get(precision, np.float32)
    
    def _init_tensorrt(self) -> None:
        """Initialize TensorRT engine and context."""
        self.runtime = trt.Runtime(self.logger)
        
        # Initialize plugins
        trt.init_libnvinfer_plugins(None, "")
        
        # Load engine
        assert os.path.exists(self.model_path), f"Engine file not found: {self.model_path}"
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Set up I/O bindings
        self._setup_io_bindings()
        
    def _setup_io_bindings(self) -> None:
        """Set up input/output bindings for TensorRT engine."""
        self.outputs = []
        self.bindings = []
        
        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            shape = self.engine.get_tensor_shape(binding_name)
            size = trt.volume(shape)
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding_idx):
                self.input = HostDeviceMem(host_mem, device_mem)
                self.input_shape = shape
                self.input_dtype = dtype
                self.context.set_binding_shape(binding_idx, shape)
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
                self.output_shape = shape
        
        self.stream = cuda.Stream()
        
        assert len(self.outputs) > 0
        assert len(self.bindings) > 0
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data.
        
        Args:
            input_data: Preprocessed input data as numpy array
            
        Returns:
            List of output arrays from model inference
        """
        # Copy input data to host buffer
        np.copyto(self.input.host, input_data.ravel())
        
        # Run inference
        self.cuda_ctx.push()
        
        # Copy data to device
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        # Copy results back to host
        predictions = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            predictions.append(out.host.reshape(self.output_shape))
            
        self.stream.synchronize()
        self.cuda_ctx.pop()
        
        return predictions
    
    def __del__(self):
        """Cleanup CUDA context."""
        if hasattr(self, 'cuda_ctx'):
            self.cuda_ctx.pop()

#!/bin/bash

# Get the root directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..") # Navigate to the root directory of the project
MODEL_DIR="$ROOT_DIR/model"
CONFIG_FILE="$ROOT_DIR/inference_ros2/config/default_params.yaml"

# Default values
MODEL_NAME="yolov8-keypoint-det"
DEFAULT_ONNX_PATH="$MODEL_DIR/800.onnx"
PLATFORMS=("nuc" "custom" "rtx-3070")
PRECISIONS=("fp32" "fp16" "int8")
DEFAULT_PRECISION="fp32"

# Function to retrieve TensorRT version
get_tensorrt_version() {
    python3 -c "import tensorrt; print(tensorrt.__version__)"
}

# Prompt for model name
read -p "Enter model name (Press Enter to use default) [default: $MODEL_NAME]: " INPUT_MODEL_NAME
MODEL_NAME=${INPUT_MODEL_NAME:-$MODEL_NAME}

# Prompt for platform
echo "Select platform:"
select PLATFORM in "${PLATFORMS[@]}"; do
    if [[ " ${PLATFORMS[@]} " =~ " ${PLATFORM} " ]]; then
        break
    else
        echo "Invalid selection. Please choose a valid platform."
    fi
done

# Prompt for ONNX file path
read -p "Enter ONNX file path (Press Enter to use default) [default: $DEFAULT_ONNX_PATH]: " INPUT_ONNX_PATH
ONNX_PATH=${INPUT_ONNX_PATH:-$DEFAULT_ONNX_PATH}

# Prompt for precision
read -p "Enter precision (fp32, fp16, int8) (Press Enter to use default) [default: $DEFAULT_PRECISION]: " INPUT_PRECISION
PRECISION=${INPUT_PRECISION:-$DEFAULT_PRECISION}

# Validate precision input
if [[ ! " ${PRECISIONS[@]} " =~ " ${PRECISION} " ]]; then
    echo "Invalid precision selected. Defaulting to $DEFAULT_PRECISION."
    PRECISION=$DEFAULT_PRECISION
fi

# Set trtexec flags based on precision
TRTEXEC_FLAGS=""
if [[ "$PRECISION" == "fp16" ]]; then
    TRTEXEC_FLAGS="--fp16"
elif [[ "$PRECISION" == "int8" ]]; then
    TRTEXEC_FLAGS="--int8"
fi

# Retrieve TensorRT version
TENSORRT_VERSION=$(get_tensorrt_version)

# Export to TensorRT
ENGINE_NAME="$MODEL_DIR/${MODEL_NAME}-${PLATFORM}-${PRECISION}-${TENSORRT_VERSION}-800.engine"
trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_NAME --allowGPUFallback --memPoolSize=workspace:5000 $TRTEXEC_FLAGS

# Update the YAML configuration file
if [[ -f "$CONFIG_FILE" ]]; then
    # Use sed to update the model_path entry
    sed -i "s|model_path: .*|model_path: \"$ENGINE_NAME\"|" "$CONFIG_FILE"
    if grep -q "model_path: \"$ENGINE_NAME\"" "$CONFIG_FILE"; then
        echo "Updated model_path in $CONFIG_FILE to: $ENGINE_NAME"
    else
        echo "Failed to update model_path in $CONFIG_FILE. Please check the file format."
    fi
else
    echo "Configuration file not found: $CONFIG_FILE"
fi

echo "Exported ONNX model to TensorRT engine: $ENGINE_NAME"
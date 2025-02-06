#!/bin/bash
export HF_HOME="/DATA/disk0/nby/xwl/.cache/huggingface"
export TRANSFORMERS_CACHE=$HF_HOME

# Define paths
INFERENCE_SCRIPT="/home/nby/xwl/Unify_Benchmark/inference/inference.py"
INPUT_FILE="/home/nby/xwl/Unify_Benchmark/Unify_Dataset/Understanding/Understanding_Unified_copy.json"
OUTPUT_DIR="/home/nby/xwl/Unify_Benchmark/results/"
MODEL_ID="emu3"

# Create cache directory if it doesn't exist
mkdir -p "/DATA/disk0/nby/xwl/.cache/huggingface/hub"

# Add Emu3 module path to PYTHONPATH
export PYTHONPATH="/DATA/disk0/nby/xwl/model/Emu3:$PYTHONPATH"

# Set HF Mirror
export HF_ENDPOINT="https://hf-mirror.com"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "Starting inference with $MODEL_ID..."
echo "Using custom cache directory: $HF_HOME"

# Run inference script
python "$INFERENCE_SCRIPT" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR" \
    --model_id "$MODEL_ID"

echo "Inference completed. Results saved in $OUTPUT_DIR"
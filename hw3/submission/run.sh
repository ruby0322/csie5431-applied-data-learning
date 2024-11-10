#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <base_model_path> <adapter_checkpoint_path> <input_file> <output_file>"
    echo "Example:"
    echo "bash run.sh \\"
    echo "    /path/to/zake7749/gemma-2-2b-it-chinese-kyara-dpo \\"
    echo "    /path/to/adapter_checkpoint/under/your/folder \\"
    echo "    /path/to/input \\"
    echo "    /path/to/output"
    exit 1
fi

# Store arguments in descriptive variables
BASE_MODEL_PATH="$1"
ADAPTER_PATH="$2"
INPUT_FILE="$3"
OUTPUT_FILE="$4"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist"
    exit 1
fi

# Check if adapter path exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter checkpoint directory '$ADAPTER_PATH' does not exist"
    exit 1
fi

# Run the prediction script
echo "Starting prediction..."
echo "Base model: $BASE_MODEL_PATH"
echo "Adapter: $ADAPTER_PATH"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

python3 inference.py \
    --base_model_name "$BASE_MODEL_PATH" \
    --peft_model_path "$ADAPTER_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

# Check if the prediction was successful
if [ $? -eq 0 ]; then
    echo "Prediction completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Error: Prediction failed"
    exit 1
fi
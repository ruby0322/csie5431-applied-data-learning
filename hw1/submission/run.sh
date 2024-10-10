#!/bin/bash

# Check if all three parameters are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 path_to_context.json path_to_test.json path_to_output_prediction.csv"
    exit 1
fi

# Assign command-line parameters to variables
context_json="$1"
test_json="$2"
output_prediction_csv="$3"

# Step 1: Transform the test data using the provided test.json and context.json
python transform_test_data.py "$test_json" "$context_json"

# Step 2: Run the multiple choice inference to generate predictions
python multiple_choice_inference.py \
    --test_file test_data.json \
    --model_dir ./multiple-choice-fine-tune-96 \
    --output_file multiple-choice-predictions.json

# Step 3: Transform the data for QA input using test_data.json
python transform_qa_test_data.py test_data.json

# Step 4: Run the span extraction inference to generate the final predictions
python span_extraction_inference.py \
    --test_file qa_test_data.json \
    --model_dir ./span-extraction-fine-tune-82 \
    --output_file predictions.json \
    --batch_size 4 \
    --max_seq_length 512 \
    --output_dir ./

# Step 5: Transform the predictions.json into the desired CSV format
python transform_predictions.py --input_file predictions.json --output_file "$output_prediction_csv"

echo "Predictions saved to $output_prediction_csv"

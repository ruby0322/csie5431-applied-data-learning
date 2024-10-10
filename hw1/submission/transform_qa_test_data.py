import argparse
import json

import numpy as np
import pandas as pd


def construct_qa_input(predicted_options, test_data):
    # Initialize a list to hold QA input data
    qa_data = []

    # Extract the options from the test_data (assumes the multiple-choice options are in the keys ending0, ending1, ...)
    option_keys = [f"ending{i}" for i in range(4)]

    for i, t in enumerate(test_data):
        # Extract the options for the current example from the test_data
        predicted_options_idx = predicted_options['predictions'][i]
        options = [t[option_key] for option_key in option_keys]

        # Construct QA input: use the predicted option as the question
        qa_input = {
            "id": t['id'],
            "context": options[predicted_options_idx],
            "question": t['sent1'],
            "answers": {"text": "", "answer_start": -1}  # Placeholder for answers
        }
        qa_data.append(qa_input)

    return qa_data

def main(test_data_file_path):
    with open('multiple-choice-predictions.json', 'r') as f:
        predicted_options = json.load(f)

    with open(test_data_file_path, 'r') as f:
        test_data = json.load(f)

    qa_data = construct_qa_input(predicted_options, test_data)

    with open('qa_test_data.json', 'w') as f:
        json.dump(qa_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct QA input data from multiple-choice predictions and test data.")
    parser.add_argument('test_data_file_path', type=str, help='Path to the test data JSON file.')
    
    args = parser.parse_args()

    main(args.test_data_file_path)

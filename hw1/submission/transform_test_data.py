import argparse
import json

import numpy as np
import pandas as pd


def transform_to_testing_format(df, contexts):
    formatted_data = []

    for index, row in df.iterrows():
        question = row['question']
        paragraphs = row['paragraphs']

        entry = {
            "id": row['id'],
            "sent1": question,
            "ending0": contexts[paragraphs[0]],
            "ending1": contexts[paragraphs[1]],
            "ending2": contexts[paragraphs[2]],
            "ending3": contexts[paragraphs[3]],
        }

        formatted_data.append(entry)

    return formatted_data

def main(test_data_file_path, context_data_file_path):
    contexts_df = pd.read_json(context_data_file_path)
    contexts = np.array(contexts_df[0])

    test = pd.read_json(test_data_file_path)
    test_data = transform_to_testing_format(test, contexts)

    with open('test_data.json', 'w') as test_data_json:
        json.dump(test_data, test_data_json)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform test and context data to testing format.")
    parser.add_argument('test_data_file_path', type=str, help='Path to the test data JSON file.')
    parser.add_argument('context_data_file_path', type=str, help='Path to the context data JSON file.')
    
    args = parser.parse_args()

    main(args.test_data_file_path, args.context_data_file_path)

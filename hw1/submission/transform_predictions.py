import csv
import json

input_json_file = 'predictions.json'
output_csv_file = 'predictions.csv'

with open(input_json_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

csv_columns = ['id', 'prediction_text']

with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)


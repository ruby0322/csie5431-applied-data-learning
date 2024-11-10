import argparse
import re

import numpy as np
import pandas as pd

TASK_TO_ACCIENT = '翻譯成文言文'
TASK_TO_MODERN = '翻譯成白話文'

INSTRUCTIONS = [
    '古代怎麼說',
    '現代怎麼說',
    '翻譯成文言文',
    '翻譯成古文',
    '翻譯成現代文',
    '翻譯成白話文',
    '文言文翻譯',
    '現代文翻譯',
]

REMOVED_TEXT_TASK_MAPPING = {
    '這句話在古代怎麼說': TASK_TO_ACCIENT,
    '這句話在現代怎麼說': TASK_TO_MODERN,
    '翻譯成文言文': TASK_TO_ACCIENT,
    '翻譯成古文': TASK_TO_ACCIENT,
    '翻譯成現代文': TASK_TO_MODERN,
    '翻譯成白話文': TASK_TO_MODERN,
    '文言文翻譯': TASK_TO_MODERN,
    '現代文翻譯': TASK_TO_ACCIENT,
}

pattern = '|'.join(INSTRUCTIONS)

# 篩選 rows 並移除指令周圍的文本
def filter_and_remove_text(row):
    instruction_text = row['instruction']
    match = re.search(pattern, instruction_text)
    if match:
        start, end = match.span()
        before_text = re.split(r'[，。！？：；]', instruction_text[:start])[-1]
        after_text = re.split(r'[，。！？：；]', instruction_text[end:])[0]
        new_instruction = instruction_text.replace(before_text + match.group() + after_text, '').strip()
        new_instruction = re.sub(r'^[，。！？：；]', '', new_instruction).strip()
        new_instruction = re.sub(r'答案：', '', new_instruction).strip()
        new_instruction = re.sub(r'。。', '。', new_instruction).strip()
        new_instruction = re.sub(r'[:：]$', '', new_instruction).strip()
        removed_text = (before_text + match.group() + after_text).strip()
        task = ''
        for k, v in REMOVED_TEXT_TASK_MAPPING.items():
            if k in removed_text:
                task = v
                break
        return pd.Series([new_instruction, task])
    return pd.Series([instruction_text, ''])

def preprocess(df):
    classification = df.apply(filter_and_remove_text, axis=1)
    classification_df = pd.DataFrame(classification)
    preprocessed_df = pd.DataFrame(df)
    preprocessed_df['instruction'] = classification_df[0]
    preprocessed_df['task'] = classification_df[1]
    # preprocessed_df = preprocessed_df.dropna().reset_index(drop=True)
    preprocessed_df['instruction'] = np.where(
        preprocessed_df['task'] != '',
        preprocessed_df['instruction'] + preprocessed_df['task'] + '：',
        preprocessed_df['instruction']
    )
    # preprocessed_df = preprocessed_df.drop(columns=['task'])
    return preprocessed_df

# Function to load, preprocess, and save data
def preprocess_and_save(input_filename, output_filename):
    df = pd.read_json(input_filename)
    preprocessed_df = preprocess(df)
    preprocessed_df.to_json(output_filename, index=False, orient="records")
    print(f"Data has been saved to {output_filename}")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save text transformations.")
    parser.add_argument("input_filename", type=str, help="Input JSON filename with data to process")
    parser.add_argument("output_filename", type=str, help="Output JSON filename to save processed data")
    
    args = parser.parse_args()
    
    preprocess_and_save(args.input_filename, args.output_filename)

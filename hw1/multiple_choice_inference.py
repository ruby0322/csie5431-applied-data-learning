import argparse
import json

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on a multiple choice task using a fine-tuned model")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the fine-tuned model directory", required=True)
    parser.add_argument("--test_file", type=str, help="Path to the test file (csv/json format)", required=True)
    parser.add_argument("--output_file", type=str, help="Path to save the predictions", required=True)
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max input sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()
    return args

def preprocess_function(examples, tokenizer, max_seq_length):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    second_sentences = [[examples[f"ending{i}"][idx] for i in range(4)] for idx in range(len(examples["sent1"]))]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_seq_length, padding="max_length")
    
    # Unflatten the tokenized inputs so that each question has 4 choices
    tokenized_inputs = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return tokenized_inputs

def main():
    args = parse_args()

    # Load tokenizer, config, and model from the saved directory (output_dir)
    config = AutoConfig.from_pretrained(args.model_name_or_path)  # Load configuration
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config=config)
    
    model.eval()  # Set model to evaluation mode
    
    # Load the test dataset (assuming it's a dataset similar to SWAG or other multiple choice dataset)
    test_dataset = load_dataset('json', data_files={'test': args.test_file})['test']
    
    # Preprocess the test dataset
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer, args.max_seq_length), batched=True)
    
    # DataLoader for test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []

    # Perform inference
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # Save the predictions to the output file
    with open(args.output_file, 'w') as f:
        json.dump(all_predictions, f)
    
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()

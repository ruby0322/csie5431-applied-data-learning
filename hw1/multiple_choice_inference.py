import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference with a pretrained transformers model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path")
    parser.add_argument("--eval_file", type=str, required=True, help="A csv or json file for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum input sequence length")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, help="Where to save the results")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config=config)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    data_files = {"test": args.eval_file}
    raw_datasets = load_dataset('json', data_files=data_files)
    test_dataset = raw_datasets['test']

    # Preprocessing the input data for inference
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        second_sentences = [
            [examples[f"ending{i}"][idx] for i in range(4)] for idx in range(len(examples["sent1"]))
        ]
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=args.max_seq_length, padding="max_length")
        return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size)

    model.eval()

    results = []
    for batch in tqdm(test_dataloader, desc="Performing inference"):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        results.extend(predictions)

    # Save or print the predictions
    if args.output_dir:
        output_file = Path(args.output_dir) / "inference_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f)
        print(f"Inference results saved to {output_file}")
    else:
        print("Inference results:", results)

if __name__ == "__main__":
    main()

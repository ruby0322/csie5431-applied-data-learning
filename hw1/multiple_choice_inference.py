import argparse
import json
import os
from itertools import chain

import torch
from accelerate import Accelerator
from datasets import load_dataset
from numpyencoder import NumpyEncoder
from torch.utils.data import DataLoader
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          default_data_collator)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load fine-tuned multiple choice model and make predictions on test set."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output",
        help="The directory where the fine-tuned model is stored."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="A CSV or JSON file containing the test data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_predictions.json",
        help="The file where the predictions will be saved."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation."
    )
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_seq_length):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    second_sentences = [
        [examples[f"ending{i}"][idx] for i in range(4)] for idx in range(len(examples["sent1"]))
    ]
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"
    )
    return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def load_model_and_tokenizer(model_dir):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def make_predictions(model, tokenizer, test_dataset, batch_size, max_seq_length):
    accelerator = Accelerator()
    model.eval()
    data_collator = default_data_collator

    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer, max_seq_length), batched=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    
    test_dataloader = accelerator.prepare(test_dataloader)
    model = accelerator.prepare(model)

    all_predictions = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(accelerator.gather(predictions).cpu().numpy())

    return all_predictions

def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Load test dataset
    data_files = {"test": args.test_file}
    extension = args.test_file.split(".")[-1]
    test_dataset = load_dataset(extension, data_files=data_files)["test"]

    # Make predictions
    predictions = make_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        batch_size=args.per_device_eval_batch_size,
        max_seq_length=args.max_seq_length
    )

    # Save predictions
    test_ids = test_dataset["id"]
    with open(args.output_file, "w") as f:
        json.dump({"id": test_ids, "predictions": predictions}, f, indent=4, sort_keys=True,
            separators=(', ', ': '), ensure_ascii=False,
            cls=NumpyEncoder)
    
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()

import argparse
import json
from itertools import chain

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          EvalPrediction, default_data_collator)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load fine-tuned multiple choice model and evaluate on validation set."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output",
        help="Directory where the fine-tuned model is stored."
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        required=True,
        help="Path to the validation data (CSV or JSON format)."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.json",
        help="File where the evaluation results will be saved."
    )
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_seq_length):
    ending_names = [f"ending{i}" for i in range(4)]
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    second_sentences = [
        [examples[end][i] for end in ending_names] for i, _ in enumerate(examples["sent1"])
    ]
    
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    tokenized_inputs = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

def load_model_and_tokenizer(model_dir):
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate_model(model, tokenizer, validation_dataset, batch_size, max_seq_length):
    data_collator = default_data_collator
    validation_dataset = validation_dataset.map(lambda x: preprocess_function(x, tokenizer, max_seq_length), batched=True)
    validation_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)

    metric = evaluate.load("accuracy")
    model.eval()

    all_predictions = []
    all_references = []

    for batch in tqdm(validation_dataloader):
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != "labels"})
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(batch["labels"].cpu().numpy())

    accuracy = metric.compute(predictions=all_predictions, references=all_references)
    return accuracy

def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Load validation dataset
    extension = args.validation_file.split(".")[-1]
    validation_dataset = load_dataset(extension, data_files={"validation": args.validation_file})["validation"]

    # Evaluate the model
    accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        validation_dataset=validation_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )

    # Save evaluation results
    with open(args.output_file, "w") as f:
        json.dump(accuracy, f)

    print(f"Evaluation results saved to {args.output_file}")

if __name__ == "__main__":
    main()

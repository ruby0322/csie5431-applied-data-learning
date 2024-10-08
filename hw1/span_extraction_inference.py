import argparse
import json

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          default_data_collator)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load fine-tuned question-answering model and make predictions on test set."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output",
        help="Directory where the fine-tuned model is stored."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test data (CSV or JSON format)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_predictions.json",
        help="File where the predictions will be saved."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="Maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_seq_length):
    # Clean questions by stripping leading whitespace
    questions = [q.lstrip() for q in examples["question"]]

    # Tokenize questions and contexts with possible overflow handling
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",  # Ensures all sequences are padded to the same length
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    # Map the overflowed tokens back to their original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    # Adjust the offset mapping and associate the tokenized examples with their original IDs
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if tokenizer.padding_side == "right" else 0

        sample_index = sample_mapping[i]  # Find the original example this token belongs to
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Adjust offset mapping, keeping context positions and discarding others
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def load_model_and_tokenizer(model_dir):
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def postprocess_predictions(examples, features, start_logits, end_logits):
    # Post-process predictions
    all_predictions = []
    for i in range(len(start_logits)):
        start_index = torch.argmax(start_logits[i]).item()
        end_index = torch.argmax(end_logits[i]).item()
        prediction = features["offset_mapping"][i][start_index:end_index + 1]
        pred_text = examples["context"][i][prediction[0][0]:prediction[-1][1]]
        all_predictions.append({"id": features["example_id"][i], "prediction_text": pred_text})
    return all_predictions

def make_predictions(model, tokenizer, test_dataset, batch_size, max_seq_length):
    data_collator = default_data_collator
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer, max_seq_length), batched=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size)
    
    all_start_logits = []
    all_end_logits = []
    
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != "example_id"})
            all_start_logits.append(outputs.start_logits.cpu())
            all_end_logits.append(outputs.end_logits.cpu())
    
    start_logits = torch.cat(all_start_logits, dim=0)
    end_logits = torch.cat(all_end_logits, dim=0)

    predictions = postprocess_predictions(test_dataset, batch, start_logits, end_logits)
    return predictions

def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Load test dataset
    extension = args.test_file.split(".")[-1]
    test_dataset = load_dataset(extension, data_files={"test": args.test_file})["test"]

    # Make predictions
    predictions = make_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )

    # Save predictions
    with open(args.output_file, "w") as f:
        json.dump(predictions, f)
    
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()

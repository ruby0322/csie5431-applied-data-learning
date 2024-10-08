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

    # Tokenize questions and contexts, ensuring truncation and padding
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
        max_length=max_seq_length,
        padding="max_length",  # Ensures all sequences are padded to the same length
        stride=128,
        return_overflowing_tokens=True,  # Handle cases where the context exceeds max_seq_length
        return_offsets_mapping=True,  # Keep track of token offsets for later processing
    )

    # Map the overflowed tokens back to their original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    # Adjust the offset mapping and associate the tokenized examples with their original IDs
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if tokenizer.padding_side == "right" else 0

        # Ensure that all tokenized sequences are exactly of max length (for pyarrow compatibility)
        if len(tokenized_examples["input_ids"][i]) != max_seq_length:
            raise ValueError(
                f"Expected sequence length {max_seq_length}, but got {len(tokenized_examples['input_ids'][i])}"
            )

        # Map this tokenized example to the corresponding original example
        sample_index = sample_mapping[i]
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

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    column_names = test_dataset.column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

     # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    test_dataset = test_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
    )

    data_collator = default_data_collator
    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataloader = DataLoader(
        test_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_test_batch_size
    )

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat


    all_start_logits = []
    all_end_logits = []

    model.eval()

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

    print(all_start_logits)
    print(all_end_logits)

    # # Make predictions
    # predictions = make_predictions(
    #     model=model,
    #     tokenizer=tokenizer,
    #     test_dataset=test_dataset,
    #     batch_size=args.batch_size,
    #     max_seq_length=min(args.max_seq_length, tokenizer.model_max_length)
    # )

    # Save predictions
    # with open(args.output_file, "w") as f:
    #     json.dump(predictions, f)
    
    # print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()

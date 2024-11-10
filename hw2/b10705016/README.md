# Training a Seq2Seq Summarization Model

This repository contains a script for training a sequence-to-sequence (seq2seq) summarization model using the Hugging Face Transformers library. The model can be fine-tuned on custom datasets for text summarization tasks.

## Table of Contents

- [Training a Seq2Seq Summarization Model](#training-a-seq2seq-summarization-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
    - [1. Prepare the Environment](#1-prepare-the-environment)
    - [2. Configure the Script](#2-configure-the-script)
    - [3. Run the Training Script](#3-run-the-training-script)
  - [Evaluating the Model](#evaluating-the-model)
  - [Generating Summaries](#generating-summaries)
  - [References](#references)

## Overview

This script allows you to fine-tune pre-trained transformer models for text summarization tasks. It supports:

- Custom datasets in JSON or CSV format.
- Data preprocessing and tokenization.
- Training with configurable hyperparameters.
- Evaluation using ROUGE metrics.
- Prediction and generation of summaries.

The script leverages the Hugging Face Transformers library, along with the Datasets library for handling data and evaluation.

## Prerequisites

- Python 3.6 or higher
- Install the required packages:

  ```bash
  pip install transformers datasets nltk tw_rouge pandas
  ```

- **Note**: The script uses NLTK for sentence tokenization and requires downloading specific NLTK data files.

  ```python
  import nltk
  nltk.download('punkt')
  ```

## Dataset Preparation

Your dataset should be JSONL format and contain at least two columns:

- **Text Column**: Contains the input texts (documents) to be summarized.
- **Summary Column**: Contains the target summaries for the input texts.

**Example JSON Lines Format (`data.jsonl`):**

```json
{"text": "This is the first document to summarize.", "summary": "First summary."}
{"text": "This is the second document to summarize.", "summary": "Second summary."}
...
```

**Dataset Splits:**

- **Training Set**: Used for training the model.
- **Validation Set**: Used for evaluating the model during training.
- **Test Set**: Used for final evaluation and generating predictions.

Ensure that your data files are properly formatted and paths are correctly specified when running the script.

## Training the Model

### 1. Prepare the Environment

- Import necessary libraries and ensure that NLTK data files are downloaded.

  ```python
  import nltk
  nltk.download('punkt')
  ```

### 2. Configure the Script

You can configure the training by setting arguments either through the command line or by creating a JSON configuration file.

**Command Line Arguments:**

- **Model Parameters:**

  - `--model_name_or_path`: Path or identifier of the pre-trained model (e.g., `t5-small`, `facebook/bart-base`).
  - `--config_name`: Optional configuration name if different from the model name.
  - `--tokenizer_name`: Optional tokenizer name if different from the model name.

- **Data Parameters:**

  - `--train_file`: Path to the training data file.
  - `--validation_file`: Path to the validation data file.
  - `--test_file`: Path to the test data file (optional).
  - `--text_column`: Name of the text column in your dataset.
  - `--summary_column`: Name of the summary column in your dataset.
  - `--max_source_length`: Maximum length of the input sequences.
  - `--max_target_length`: Maximum length of the target summaries.
  - `--pad_to_max_length`: Whether to pad sequences to the maximum length.

- **Training Parameters:**

  - `--output_dir`: Directory where the model checkpoints and outputs will be saved.
  - `--per_device_train_batch_size`: Batch size per device during training.
  - `--per_device_eval_batch_size`: Batch size per device during evaluation.
  - `--learning_rate`: Learning rate for the optimizer.
  - `--num_train_epochs`: Number of training epochs.
  - `--save_steps`: Save checkpoint every specified number of steps.
  - `--seed`: Random seed for reproducibility.

### 3. Run the Training Script

**Example Command:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python summarization_train.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --text_column maintext \
    --summary_column title \
    --output_dir ./output/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --report_to wandb \
    --run_name adl-hw2 \
    --max_source_length 512 \
    --max_target_length 64 \
    --max_eval_samples 512 \
    --num_beans 4
```

**Notes:**

- Replace `train_summarization.py` with the name of your script file.
- Adjust hyperparameters like batch size and learning rate based on your resources.
- Ensure that the `text_column` and `summary_column` match the column names in your dataset.

## Evaluating the Model

During training, the script will evaluate the model on the validation set after each epoch or at specified intervals.

- **Metrics Used:** ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
- **Evaluation Configuration:**
  - Set `--do_eval` to enable evaluation.
  - Use `--evaluation_strategy` to specify when to evaluate (e.g., `steps`, `epoch`).

**Example:**

```bash
--do_eval \
--evaluation_strategy epoch \
```

The evaluation results will be logged and saved in the `output_dir`.

## Generating Summaries

After training, you can use the script to generate summaries on the test set.

- **Generate Predictions:**

  ```bash
  --do_predict \
  --test_file test_data.jsonl \
  --output_file predictions.jsonl \
  ```

- **Output:**

  - The generated summaries will be saved in the specified `output_file`.
  - A text file `generated_predictions.txt` containing the summaries will also be saved in the `output_dir`.
  - A json lines file named `output_file` will also be created.

**Example Command:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python summarization_train.py \
    --model_name_or_path ./checkpoint-16284 \
    --do_predict \
    --test_file ./data/sample_test.jsonl \
    --source_prefix "summarize: " \
    --text_column maintext \
    --output_dir ./preds/ \
    --output_file ./output.jsonl \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_beams 4
```

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Seq2Seq Training Examples](https://huggingface.co/docs/transformers/training#seq2seq-training)
- [Huggingface Summarization Trainer Example](https://github.com/huggingface/transformers/blob/t5-fp16-no-nans/examples/pytorch/summarization/run_summarization.py)

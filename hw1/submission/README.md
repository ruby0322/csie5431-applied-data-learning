# Training Transformer Models for Multiple-Choice and Span Extraction Tasks

This repository contains scripts to fine-tune transformer models for both multiple-choice and span extraction (question-answering) tasks using the Hugging Face Transformers and Accelerate libraries. These scripts can be adapted to various datasets and tasks requiring contextual understanding and answer extraction.

## Table of Contents

- [Training Transformer Models for Multiple-Choice and Span Extraction Tasks](#training-transformer-models-for-multiple-choice-and-span-extraction-tasks)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Training the Multiple-Choice Model](#training-the-multiple-choice-model)
    - [1. Preparing the Environment](#1-preparing-the-environment)
    - [2. Parsing Arguments](#2-parsing-arguments)
    - [3. Loading and Preprocessing the Dataset](#3-loading-and-preprocessing-the-dataset)
    - [4. Loading the Model and Tokenizer](#4-loading-the-model-and-tokenizer)
    - [5. Creating Data Loaders](#5-creating-data-loaders)
    - [6. Setting Up the Optimizer and Scheduler](#6-setting-up-the-optimizer-and-scheduler)
    - [7. Training Loop](#7-training-loop)
    - [8. Evaluation](#8-evaluation)
    - [9. Saving the Model](#9-saving-the-model)
  - [Training the Span Extraction Model](#training-the-span-extraction-model)
    - [1. Preparing the Environment](#1-preparing-the-environment-1)
    - [2. Parsing Arguments](#2-parsing-arguments-1)
    - [3. Loading and Preprocessing the Dataset](#3-loading-and-preprocessing-the-dataset-1)
    - [4. Loading the Model and Tokenizer](#4-loading-the-model-and-tokenizer-1)
    - [5. Creating Data Loaders](#5-creating-data-loaders-1)
    - [6. Setting Up the Optimizer and Scheduler](#6-setting-up-the-optimizer-and-scheduler-1)
    - [7. Training Loop](#7-training-loop-1)
    - [8. Evaluation](#8-evaluation-1)
    - [9. Saving the Model](#9-saving-the-model-1)
  - [Combining the Models for Paragraph Selection and Answer Extraction](#combining-the-models-for-paragraph-selection-and-answer-extraction)
  - [Training the Models](#training-the-models)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Training Scripts](#2-training-scripts)
  - [References](#references)

## Overview

- **Multiple-Choice Model**: Fine-tunes a transformer model to select the correct paragraph given a question.
- **Span Extraction Model**: Fine-tunes a transformer model to extract the answer from a given paragraph and question.
- **Combined Task**: Uses the multiple-choice model to select the most relevant paragraph and the span extraction model to find the answer within that paragraph.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers library
- Hugging Face Datasets library
- Accelerate library
- Additional packages: `datasets`, `evaluate`, `numpy`, `argparse`, etc.

Install the required packages using:

```bash
pip install torch transformers datasets accelerate evaluate numpy
```

## Training the Multiple-Choice Model

### 1. Preparing the Environment

- **Import Libraries**: The script imports necessary libraries like `argparse`, `datasets`, `transformers`, `torch`, and `accelerate`.
- **Initialize Accelerator**: An `Accelerator` object is created to handle device placement and distributed training.

### 2. Parsing Arguments

- The script uses `argparse` to define command-line arguments for customization.
- Key arguments include dataset paths, model configurations, training hyperparameters, and output directories.

### 3. Loading and Preprocessing the Dataset

- **Dataset Loading**: The dataset is loaded either from the Hugging Face Hub or local files.
- **Data Structure**: For multiple-choice tasks, each example should include:
  - A question or context (`sent1`).
  - Multiple choices (`ending0`, `ending1`, etc.).
  - A label indicating the correct choice.
- **Preprocessing**:
  - The `preprocess_function` tokenizes the inputs.
  - It creates pairs of context and choices for the model.
  - Labels are aligned with the tokenized inputs.

### 4. Loading the Model and Tokenizer

- **Tokenizer**: Loaded using `AutoTokenizer.from_pretrained()`.
- **Model**: Loaded using `AutoModelForMultipleChoice.from_pretrained()`.
- **Vocabulary Adjustment**: If the tokenizer's vocabulary size differs from the model's, the model embeddings are resized.

### 5. Creating Data Loaders

- **Data Collator**: A custom `DataCollatorForMultipleChoice` handles dynamic padding and batching.
- **Data Loaders**: Training and evaluation data loaders are created with appropriate batch sizes.

### 6. Setting Up the Optimizer and Scheduler

- **Optimizer**: An `AdamW` optimizer is set up with weight decay for regularization.
- **Scheduler**: A learning rate scheduler is created using `get_scheduler()`.

### 7. Training Loop

- The model is trained over multiple epochs.
- **Gradient Accumulation**: If specified, gradients are accumulated over several steps before updating the model.
- **Checkpointing**: The model state can be saved at specified intervals.

### 8. Evaluation

- After each epoch, the model is evaluated on the validation set.
- **Metrics**: Accuracy is calculated using the `evaluate` library.
- **Logging**: Evaluation metrics are logged and can be tracked using experiment tracking tools if enabled.

### 9. Saving the Model

- The trained model and tokenizer are saved to the specified output directory.
- If pushing to the Hugging Face Hub is enabled, the model is uploaded.

---

## Training the Span Extraction Model

### 1. Preparing the Environment

- Similar to the multiple-choice model, necessary libraries are imported, and an `Accelerator` is initialized.

### 2. Parsing Arguments

- The script defines command-line arguments specific to question-answering tasks, such as `--doc_stride`, `--max_answer_length`, and `--version_2_with_negative`.

### 3. Loading and Preprocessing the Dataset

- **Dataset Loading**: The dataset is loaded from the Hugging Face Hub or local files.
- **Data Structure**: Each example should include:
  - A question.
  - A context paragraph.
  - Answers with text and start positions.
- **Preprocessing**:
  - The `prepare_train_features` function tokenizes and processes the training data, handling long contexts by creating sliding windows with `doc_stride`.
  - The `prepare_validation_features` function processes the validation data, keeping track of the mapping between tokens and original text for accurate evaluation.

### 4. Loading the Model and Tokenizer

- **Tokenizer**: Loaded with `AutoTokenizer.from_pretrained()`, ensuring `use_fast=True` for efficient tokenization.
- **Model**: Loaded using `AutoModelForQuestionAnswering.from_pretrained()`.

### 5. Creating Data Loaders

- **Data Collator**: Uses `DataCollatorWithPadding` for dynamic padding.
- **Data Loaders**: Created for training and evaluation datasets.

### 6. Setting Up the Optimizer and Scheduler

- Similar to the multiple-choice model, an optimizer and scheduler are set up.

### 7. Training Loop

- The model is trained over multiple epochs with gradient accumulation and optional checkpointing.

### 8. Evaluation

- **Post-processing**:
  - The `postprocess_qa_predictions` function is used to convert model outputs into human-readable answers.
  - It aligns predicted token positions with the original text.
- **Metrics**:
  - Evaluation metrics like Exact Match (EM) and F1 score are computed using the `evaluate` library.
- **Logging**: Evaluation results are logged and can be tracked.

### 9. Saving the Model

- The trained model and tokenizer are saved.
- If enabled, the model is uploaded to the Hugging Face Hub.

---

## Combining the Models for Paragraph Selection and Answer Extraction

To solve the task of selecting the correct paragraph and extracting the answer given a question, the two models are used sequentially:

1. **Paragraph Selection**:

   - **Input**: A question and multiple paragraphs.
   - **Process**:
     - Use the multiple-choice model to evaluate which paragraph is most relevant to the question.
     - The model predicts the paragraph that best answers the question.
   - **Output**: The selected paragraph.

2. **Answer Extraction**:

   - **Input**: The question and the selected paragraph from the previous step.
   - **Process**:
     - Use the span extraction (question-answering) model to find the exact answer within the selected paragraph.
     - The model predicts the start and end positions of the answer span.
   - **Output**: The extracted answer text.

3. **Transformation of Predictions**:

   - The predictions from the multiple-choice model (the index of the selected paragraph) are transformed to select the corresponding paragraph text.
   - This paragraph text is then formatted as input for the span extraction model along with the question.

4. **Final Output**:

   - The answer extracted from the selected paragraph is the final output.

By combining the two models in this way, the system first narrows down the search space by selecting the most relevant paragraph and then accurately extracts the answer from that paragraph.

---

## Training the Models

### 1. Data Preparation

To train the models, you need to ensure that your data is formatted in the expected form. If your training data comes from Kaggle or a similar source, you can use the following functions to transform the data into the required format.

**For Multiple-Choice Model:**

```python
def transform_to_training_format(df, contexts):
    formatted_data = []

    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        question = row['question']
        paragraphs = row['paragraphs']  # These are indices of contexts
        relevant = row['relevant']

        # Create the format
        entry = {
            "sent1": question,
            "ending0": contexts[paragraphs[0]],
            "ending1": contexts[paragraphs[1]],
            "ending2": contexts[paragraphs[2]],
            "ending3": contexts[paragraphs[3]],
            "label": paragraphs.index(relevant)  # Find the index of the relevant context
        }

        # Append the formatted entry
        formatted_data.append(entry)

    return formatted_data
```

**For Span Extraction Model:**

```python
def transform_to_training_format(df, contexts):
    transformed_data = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        question = row['question']
        paragraphs = row['paragraphs']
        relevant_paragraph_index = row['relevant']
        answer = row['answer']

        # Find the context for the relevant paragraph
        relevant_context = contexts[relevant_paragraph_index]

        # Add the data as a dictionary
        transformed_data.append({
            'id': idx,
            'question': question,
            'context': relevant_context,
            'answers': {
                'text': [answer['text']],
                'answer_start': [answer['start']]
            }
        })

    return transformed_data
```

**Instructions:**

- **Step 1**: Load your dataset into a Pandas DataFrame (`df`).
- **Step 2**: Prepare a list or dictionary of contexts (`contexts`) indexed appropriately.
- **Step 3**: Use the provided functions to transform your data.
  - For multiple-choice tasks, use `transform_to_training_format(df, contexts)`.
  - For span extraction tasks, use `transform_data(df, contexts)`.
- **Step 4**: Save the transformed data to JSON files (`train_data.json`, `valid_data.json`).

### 2. Training Scripts

After preparing the data, you can train the models by invoking the following scripts.

**Training the Multiple-Choice Model:**

```bash
python multiple_choice_train.py \
  --train_file train_data.json \
  --validation_file valid_data.json \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --tokenizer_name hfl/chinese-roberta-wwm-ext \
  --output_dir ./mc_output \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --pad_to_max_length \
  --lr_scheduler_type cosine \
  --seed 42
```

**Training the Span Extraction Model:**

```bash
python span_extraction_train.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --train_file train_data.json \
  --validation_file valid_data.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ./qa_output \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --pad_to_max_length \
  --lr_scheduler_type linear \
  --seed 42
```

**Notes:**

- Replace `hfl/chinese-roberta-wwm-ext` with the model name or path suitable for your language or task.
- Ensure that the `--train_file` and `--validation_file` paths point to the transformed JSON files.
- Adjust hyperparameters like `--learning_rate`, `--num_train_epochs`, and batch sizes as needed.
- The `--output_dir` specifies where the trained model and tokenizer will be saved.


## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Accelerate Library Documentation](https://huggingface.co/docs/accelerate/)
- [Evaluate Library Documentation](https://huggingface.co/docs/evaluate/)

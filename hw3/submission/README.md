# Instruction Fine-Tuning for Language Model

## Data Preparation

Place your dataset in the working directory as a JSON file, structured with fields like "instruction" and "output". Example data format:

{
  "instruction": "第二年召迴朝廷，改任著作佐郎，直史館，改任左拾遺。\n翻譯成文言文：",
  "output": "明年召還，改著作佐郎，直史館，改左拾遺。"
}


## Training the Model

To train the model, simply open and run train.ipynb. The Jupyter notebook contains all the necessary setup, configuration, and training code. All dependencies are installed automatically, and the workflow is fully integrated, so you don’t need to manually adjust any settings or parameters.

Just run all cells in train.ipynb, and the model will begin training with the predefined configurations and dataset.

## Training Workflow

The training process in the script follows a streamlined workflow using `SFTTrainer`, which integrates Hugging Face’s `transformers` and LoRA-based fine-tuning (`peft`). Here’s the detailed step-by-step breakdown:

1. **Initialize the Model and Tokenizer**
   - The script loads a pre-trained language model (`AutoModelForCausalLM`) and its associated tokenizer using `from_pretrained()`.
   - LoRA configurations are applied using `prepare_model_for_kbit_training()` to enable parameter-efficient training with low-rank adaptations.

2. **Dataset Loading and Preprocessing**
   - The dataset is loaded using `datasets.load_dataset()`.
   - Input-output pairs are expected with fields like `"instruction"` and `"output"`.
   - A formatting function may be used to concatenate the prompt (`instruction`) and response (`output`) into a single training example, with appropriate special tokens added (e.g., `<bos>`, `<eos>`).

3. **LoRA Configuration**
   - LoRA-specific parameters (`rank`, `alpha`, `dropout`) are set up in `LoraConfig`.
   - The model is wrapped with LoRA configurations using `get_peft_model()`, reducing the number of trainable parameters while retaining expressive power.

4. **Setup `SFTTrainer`**
   - The `SFTTrainer` is initialized with the model, training dataset, and hyperparameters.
   - Training arguments (e.g., batch size, learning rate, epochs) are specified using `TrainingArguments`.
   - `SFTTrainer` automatically handles input-output pair formatting and tokenization.

5. **Start Training**
   - The training process begins with a call to `trainer.train()`.
   - During training, the model learns to generate responses based on the provided instructions using supervised fine-tuning.
   - The loss is calculated as the cross-entropy loss between the model’s predicted logits and the actual token labels.

6. **Evaluation (Optional)**
   - If an evaluation dataset is provided, `SFTTrainer` periodically evaluates the model using a custom `compute_metrics` function.
   - Perplexity is commonly used as a metric for language models, calculated based on the average cross-entropy loss.

7. **Checkpointing**
   - The script saves checkpoints at specified intervals (e.g., at the end of each epoch).
   - These checkpoints include model weights, optimizer states, and training metadata, allowing for resuming or further fine-tuning later.

8. **Model Saving**
   - After training, the final model is saved to the `output_dir`.
   - The trained model can then be loaded for inference or additional fine-tuning.

## Reference

- [FINE TUNING GEMMA 2B](https://github.com/benitomartin/peft-gemma-2b)
- [Fine-Tuning GEMMA-2b for Binary Classification (4-bit Quantization)](https://medium.com/@sabaybiometzger/fine-tuning-gemma-2b-for-binary-classification-4-bit-quantization-60437e877723)
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
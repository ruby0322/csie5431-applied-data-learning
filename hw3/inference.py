import argparse
import json

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def get_bnb_config():
    """Configure quantization for model loading"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_fp32_cpu_offload=True
    )

def get_prompt(instruction: str) -> str:
    """Format the instruction into a prompt template"""
    return f"你是一位精通古今中文的翻譯專家。只回覆翻譯結果，不可有多餘說明或解釋。\nUSER: {instruction}\nASSISTANT:"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate predictions using a fine-tuned model')
    parser.add_argument(
        '--base_model_name',
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
        help='Base model name or path'
    )
    parser.add_argument(
        '--peft_model_path',
        type=str,
        required=True,
        help='Path to the PEFT model'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save the predictions JSON file'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for text generation'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum number of new tokens to generate'
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize tokenizer
    print(f"Loading tokenizer from {args.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with quantization
    print(f"Loading base model from {args.base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=get_bnb_config(),
        device_map="auto"
    )
    
    # Load the PEFT model
    print(f"Loading PEFT model from {args.peft_model_path}...")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model.eval()
    
    # Read test data
    print(f"Reading test data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Generate predictions
    predictions = []
    print("Generating predictions...")
    
    for item in tqdm(test_data):
        prompt = get_prompt(item["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (after ASSISTANT:)
        response = generated_text.split("ASSISTANT:")[-1].strip()
        
        predictions.append({
            "id": item["id"],
            "output": response
        })
    
    # Save predictions
    print(f"Saving predictions to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
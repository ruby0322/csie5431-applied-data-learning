import argparse
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def get_bnb_config() -> BitsAndBytesConfig:
    """Configure quantization for model loading"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_fp32_cpu_offload=True
    )

class TranslationChatbot:
    def __init__(self, base_model_name: str, peft_model_path: str):
        """Initialize the chatbot with the specified models"""
        print("\nInitializing chatbot...")
        print(f"Loading tokenizer from {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model from {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=get_bnb_config(),
            device_map="auto"
        )
        
        print(f"Loading PEFT model from {peft_model_path}")
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model.eval()
        # Add default prompt template
        self.prompt_template = "你是一位精通古今中文的翻譯專家。只回覆翻譯結果，不可有多餘說明或解釋。\nUSER: {}\nASSISTANT:"
        print("Initialization complete!\n")
    
    def set_prompt_template(self, new_template: str) -> None:
        """Update the prompt template"""
        self.prompt_template = new_template + '\nUSER: {}\nASSISTANT:"'
        print("Prompt template updated successfully")
    
    def get_prompt(self, instruction: str) -> str:
        """Format the instruction using the current prompt template"""
        return self.prompt_template.format(instruction)
    
    def generate_response(
        self,
        instruction: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512
    ) -> str:
        """Generate a response for the given instruction"""
        prompt = self.get_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("ASSISTANT:")[-1].strip()

def print_welcome_message():
    """Print welcome message and instructions"""
    print("\n=== Chinese Translation Chatbot ===")
    print("Type your text to translate")
    print("Commands:")
    print("  /quit or /exit - Exit the chatbot")
    print("  /help         - Show this help message")
    print("  /temp <0-1>   - Set temperature (default: 0.7)")
    print("  /topp <0-1>   - Set top_p (default: 0.9)")
    print("  /set_prompt   - Update prompt template (use {} as input placeholder)")
    print("================================\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Interactive Translation Chatbot')
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize chatbot
    chatbot = TranslationChatbot(args.base_model_name, args.peft_model_path)
    
    # Initialize generation parameters
    temperature = 0.7
    top_p = 0.9
    
    print_welcome_message()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == '/help':
                print_welcome_message()
                continue
            
            elif user_input.startswith('/temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0 <= new_temp <= 1:
                        temperature = new_temp
                        print(f"Temperature set to {temperature}")
                    else:
                        print("Temperature must be between 0 and 1")
                except:
                    print("Invalid temperature value")
                continue
            
            elif user_input.startswith('/topp '):
                try:
                    new_topp = float(user_input.split()[1])
                    if 0 <= new_topp <= 1:
                        top_p = new_topp
                        print(f"Top-p set to {top_p}")
                    else:
                        print("Top-p must be between 0 and 1")
                except:
                    print("Invalid top-p value")
                continue
            
            elif user_input.startswith('/set_prompt'):
                print("Enter new prompt template (use {} as placeholder for user input):")
                new_template = input().strip()
                try:
                    chatbot.set_prompt_template(new_template)
                except ValueError as e:
                    print(f"Error: {str(e)}")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Generate response
            response = chatbot.generate_response(
                user_input,
                temperature=temperature,
                top_p=top_p
            )
            print(f"\nGuGinGemma: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main()
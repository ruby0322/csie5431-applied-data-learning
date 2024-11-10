import torch
from transformers import BitsAndBytesConfig


def get_bnb_config() -> BitsAndBytesConfig:
    """Configure quantization for QLoRA 4-bit training"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_fp32_cpu_offload=True  # Enable CPU offloading here
    )

def get_prompt(instruction: str) -> str:
    """Format the instruction into a prompt template"""
    return f"你是一位精通古今中文的翻譯專家。只回覆翻譯結果，不可有多餘說明或解釋。\nUSER: {instruction}\nASSISTANT:"

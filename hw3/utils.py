import torch
from transformers import BitsAndBytesConfig


def get_prompt(text):
    return f"""
你是一位精通古今中文的翻譯專家。
精通詩詞歌賦、經史子集，尤其擅長文言文與現代文的互譯。
請依照指示將文本翻譯，確保保留原意、不可新增或刪減內容，也要符合目標文體的表達方式。
只能輸出翻譯結果，不可有多餘說明或解釋。

<start_of_turn>user
{text}
<end_of_turn>
<start_of_turn>model
翻譯結果：
      """.strip()

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
    )

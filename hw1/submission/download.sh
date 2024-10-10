git clone https://github.com/huggingface/transformers.git
pip install -e ./transformers
pip install git+https://github.com/huggingface/accelerate
pip install datasets evaluate torch>=1.3 transformers accelerate>=0.12.0 sentencepiece!=0.1.92 protobuf tqdm numpyencoder

gdown 1NKYF5YR9RYJFXFehrHBtT2o5CsloEwOg
unzip ./multiple-choice-fine-tune-96.zip

gdown 1gY5RtQZpGafULPT6haN5LH--w-B7ZXPy
unzip ./span-extraction-fine-tune-82.zip
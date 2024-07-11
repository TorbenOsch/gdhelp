"""
Description: This script demonstrates the integration of a fine-tuned language model
for generating GDScript code snippets based on given programming problems.

Author: Torben Oschkinat - cgt104590 - Bachelor degree

Usage:
    python predict.py

Requirements:
    - The following Python libraries should be installed: torch, transformers, peft.
    - The model should be fine-tuned and saved in the specified `MODEL_PATH`.
    - The tokenizer should be saved in the specified `TOKENIZER_PATH`.

Constants:
    - MAX_SEQ_LENGTH: Maximum length of the generated sequence.
    - RETURN_TENSORS: Format of the returned tensors.
    - LOAD_IN_4BIT: Determines if the model should be loaded in 4-bit precision.
    - BNB_4BIT_USE_DOUBLE_QUANT: Configuration for double quantization.
    - BNB_4BIT_QUANT_TYPE: Type of quantization to be used.
    - BNB_4BIT_COMPUTE_DTYPE: Data type for computation.
    - DEVICE_MAP: Device mapping for model loading.
    - MODEL_PATH: Path to the fine-tuned model.
    - TOKENIZER_PATH: Path to the fine-tuned tokenizer.
    - PROMPT_TEMPLATE: Template for formatting the prompt for the model.

Variables:
    - model: The loaded language model.
    - tokenizer: The loaded tokenizer.
    - device: The device on which the model is loaded.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Constants
MAX_SEQ_LENGTH = 512
RETURN_TENSORS = "pt"
LOAD_IN_4BIT = True
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
DEVICE_MAP = "auto"
MODEL_PATH = "./fine_tuned_model"
TOKENIZER_PATH = "./fine_tuned_tokenizer"
PROMPT_TEMPLATE = """[INST] Your task is to write GDScript code to solve a programming problem.
The GDScript code must be between [GDSCRIPT] and [/GDSCRIPT] tags."""

# Load the saved tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE
)
model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE_MAP, quantization_config=quantization_config)
# Get the device on which the model is loaded
device = next(model.parameters()).device
# Define the problem prompt
prompt = "Write a function, that adds two numbers."
# Create the full prompt to be input to the model
test_prompt = f"""{PROMPT_TEMPLATE}

Problem: {prompt}
[/INST]
[GDSCRIPT]"""
# Tokenize the prompt
inputs = tokenizer(test_prompt, return_tensors=RETURN_TENSORS)
inputs = {key: value.to(device) for key, value in inputs.items()}
# Generate the model output
sample = model.generate(**inputs, max_length=MAX_SEQ_LENGTH)
# Print the input prompt and the generated output
print(f"Input: {test_prompt}")
print(f"Output:\n{tokenizer.decode(sample[0], skip_special_tokens=True)}")

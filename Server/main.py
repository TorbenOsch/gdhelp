"""
Description: This script sets up and runs an HTTP server that processes POST requests
to generate GDScript code based on input instructions.

Author: Torben Oschkinat - cgt104590 - Bachelor degree

Usage:
    python main.py

Requirements:
    - The following Python libraries should be installed: torch, transformers, peft, json, http.server.
    - The model should be fine-tuned and saved in the specified `MODEL_PATH`.
    - The tokenizer should be saved in the specified `TOKENIZER_PATH`.

Constants:
    - LOAD_IN_4BIT: Determines if the model should be loaded in 4-bit precision.
    - BNB_4BIT_USE_DOUBLE_QUANT: Configuration for double quantization.
    - BNB_4bBIT_QUANT_TYPE: Type of quantization to be used.
    - BNB_4BIT_COMPUTE_DTYPE: Data type for computation.
    - DEVICE_MAP: Device mapping for model loading.
    - MAX_LENGTH: Maximum length of the generated sequence.
    - RETURN_TENSORS: Format of the returned tensors.
    - SKIP_SPECIAL_TOKENS: Whether to skip special tokens in the output.
    - MODEL_PATH: Path to the fine-tuned model.
    - TOKENIZER_PATH: Path to the fine-tuned tokenizer.
    - PROMPT_TEMPLATE: Template for formatting the prompt for the model.
    - START_TAG: Tag indicating the start of GDScript code.
    - END_TAG: Tag indicating the end of GDScript code.

Variables:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    - device: The device on which the model is loaded.

Functions:
    - extract_gdscript_content(model_output): Extracts GDScript content from the model output.
    - create_formatted_prompt(instruction): Creates a formatted prompt for the model.
    - load_model_and_tokenizer(): Loads the tokenizer and model and sets them to global variables.
    - run(server_class=HTTPServer, handler_class=handler, port=8000): Runs an HTTP server.

Classes:
    - handler(BaseHTTPRequestHandler): HTTP request handler class for processing POST requests.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Constants
LOAD_IN_4BIT = True
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4bBIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
DEVICE_MAP = "auto"
MAX_LENGTH = 512
RETURN_TENSORS = "pt"
SKIP_SPECIAL_TOKENS = True
MODEL_PATH = "./fine_tuned_model"
TOKENIZER_PATH = "./fine_tuned_tokenizer"
PROMPT_TEMPLATE = """[INST] Your task is to write GDScript code to solve a programming problem.
The GDScript code must be between [GDSCRIPT] and [/GDSCRIPT] tags."""
START_TAG = "[GDSCRIPT]"
END_TAG = "[/GDSCRIPT]"

# Variables
model = None
tokenizer = None
device = None


def extract_gdscript_content(model_output):
    """
        Extracts GDScript content from the model output.

        Parameters
        ----------
        model_output: str
            A string containing the output from a model.

        Return
        ------
        gdscript_content: str
            A string representing the extracted GDScript code.
    """
    # Find the first and second occurrences of the start and end tags
    first_start_index = model_output.find(START_TAG)
    if first_start_index == -1:
        return "Could not generate any valuable code. Try to rephrase your Wish"

    second_start_index = model_output.find(START_TAG, first_start_index + len(START_TAG))
    if second_start_index == -1:
        return "Could not generate any valuable code. Try to rephrase your Wish"

    first_end_index = model_output.find(END_TAG)
    if first_end_index == -1:
        return "Could not generate any valuable code. Try to rephrase your Wish"

    second_end_index = model_output.find(END_TAG, first_end_index + len(END_TAG))
    if second_end_index == -1:
        return "Could not generate any valuable code. Try to rephrase your Wish"

    # Extract the content between the second set of tags
    gdscript_content = model_output[second_start_index + len(START_TAG):second_end_index]

    return gdscript_content.strip()


def create_formatted_prompt(instruction):
    """
        Creates a formatted prompt.

        Parameters
        ----------
        instruction: str
            A string containing the instruction.

        Return
        ------
        prompt: str
            A string representing the formatted prompt.
    """
    prompt =f"""{PROMPT_TEMPLATE}
    
Problem: {instruction}
[/INST]
[GDSCRIPT]"""
    return prompt


class handler(BaseHTTPRequestHandler):
    """
        HTTP request handler class for processing POST requests.

        Methods
        -------
        do_POST(self)
            Handles POST requests by reading the request data, processing it through the
            model, and sending back the generated GDScript code as a response.
    """
    def do_POST(self):
        """
                Handles POST requests by processing the input through a model and returning the output.

                Parameters
                ----------
                None

                Return
                ------
                None
            """
        self.send_response(200, "OK")
        content_length = int(self.headers['Content-Length'])

        post_data = self.rfile.read(content_length)
        json_data = json.loads(post_data.decode('utf-8'))

        formatted_prompt = create_formatted_prompt(json_data)

        inputs = tokenizer(formatted_prompt, return_tensors=RETURN_TENSORS)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        sample = model.generate(**inputs, max_length=MAX_LENGTH)

        model_output = tokenizer.decode(sample[0], skip_special_tokens=SKIP_SPECIAL_TOKENS)

        formatted_output = extract_gdscript_content(model_output)
        response_data = formatted_output
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))


def load_model_and_tokenizer():
    """
        Loads the tokenizer and model and setting them to global variables.

        Parameters
        ----------
        None

        Return
        ------
        None
    """
    # Load the saved tokenizer and model
    global tokenizer
    global model
    global device
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_4bBIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE
    )
    model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE_MAP,
                                                     quantization_config=quantization_config)

    device = next(model.parameters()).device


def run(server_class=HTTPServer, handler_class=handler, port=8000):
    """
        Runs an HTTP server.

        Parameters
        ----------
        server_class: type
            The class to be used for the HTTP server.
        handler_class: type
            The class to handle the HTTP requests.
        port: int
            The port number on which the server listens.

        Return
        ------
        None
    """
    load_model_and_tokenizer()
    print(f'Successfully loaded the model and tokenizer...')
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting http on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    """
        Entry point for running the HTTP server.

        Parameters
        ----------
        None

        Return
        ------
        None
    """
    run()

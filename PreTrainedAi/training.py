"""
Module for Fine-Tuning

This module fine-tunes a pre-trained model using a
dataset provided by huggingface. The model and tokenizer are from the HuggingFace
Transformers library, specifically designed for code generation tasks.

@Author Torben Oschkinat - cgt104590 - Bachelor degree

Requirements:
    - The following Python libraries should be installed: torch, sklearn, datasets, transformers, peft.

Constants
---------
    MODEL: str - Identifier for the pre-trained model from HuggingFace model hub.
    DATASET_NAME: str - Name of the dataset in HuggingFace datasets library.
    MAX_SEQ_LENGTH: int - Maximum sequence length for tokenization.
    TRUNCATION: bool - Whether to truncate sequences longer than MAX_SEQ_LENGTH.
    RETURN_TENSORS: str - Format for returned tensors.
    PADDING_TYPE: str - Padding type for tokenization.
    OUTPUT_DIR: str - Directory to save training outputs.
    PER_DEVICE_TRAIN_BATCH_SIZE: int - Batch size for training per GPU/device.
    EVAL_STRATEGY: str - Evaluation strategy during training.
    LEARNING_RATE: float - Learning rate for the optimizer during training.
    NUM_TRAIN_EPOCHS: int - Number of epochs for training.
    LOGGING_STEPS: int - Steps interval for logging during training.
    WEIGHT_DECAY: int - Weight decay rate for regularization during training.
    MODEL_SAVE_PATH: str - Path to save the fine-tuned model.
    TOKENIZER_SAVE_PATH: str - Path to save the fine-tuned tokenizer.
    PADDING_TOKEN: str - Token used for padding sequences.
    TEST_SIZE: float - Percentage of dataset to use as validation set during training.
    RANDOM_STATE: int - Random seed for reproducibility.
    BATCHED: bool - Whether to process the dataset in batches.
    LOAD_IN_4BIT: bool - Whether to load the model in 4-bit precision.
    BNB_4BIT_USE_DOUBLE_QUANT: bool - Whether to use double quantization for 4-bit precision.
    BNB_4BIT_QUANT_TYPE: str - Type of quantization for 4-bit precision.
    BNB_4BIT_COMPUTE_DTYPE: torch.dtype - Data type for computation in 4-bit precision.
    LORA_ALPHA: int - Hyperparameter alpha for Lora model.
    LORA_DROPOUT: float - Dropout rate for Lora model.
    LORA_R: int - Hyperparameter R for Lora model.
    LORA_BIAS: str - Bias configuration for Lora model.
    LORA_TASK_TYPE: str - Task type for Lora model.
    PROMPT_TEMPLATE: str - Template for instructions.

Functions:
    - tokenize_function(example, tokenizer): Tokenizes input examples for the model.
    - create_chat_template(example): Creates a template for instructions.
    - create_tokenized_dataset(tokenizer): Creates tokenized datasets for training and validation.
    - save_model_and_tokenizer(trainer, tokenizer): Saves the fine-tuned model and tokenizer.
    - create_trainer(model, tokenizer, data): Creates a Trainer instance for fine-tuning.
    - main(): Main function to orchestrate the fine-tuning process.
"""

import torch
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    concatenate_datasets,
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    BitsAndBytesConfig
)

# Constants - General
MODEL = 'codellama/CodeLlama-13b-Instruct-hf'
DATASET_NAME = 'dotfantasy/godot-code'
MAX_SEQ_LENGTH = 512
TRUNCATION = True
RETURN_TENSORS = 'pt'
PADDING_TYPE = 'max_length'
MODEL_SAVE_PATH = "./fine_tuned_model"
TOKENIZER_SAVE_PATH = "./fine_tuned_tokenizer"
PADDING_TOKEN = "[PAD]"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCHED = True
# Constants - TrainingArguments
OUTPUT_DIR = './results'
PER_DEVICE_TRAIN_BATCH_SIZE = 4
EVAL_STRATEGY = 'epoch'
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1000
WEIGHT_DECAY = 0
# Bitsandbytes config
LOAD_IN_4BIT = True
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
# Lora config
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_R = 16
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

PROMPT_TEMPLATE = """[INST] Your task is to write GDScript code to solve a programming problem.
The GDScript code must be between [GDSCRIPT] and [/GDSCRIPT] tags."""


def tokenize_function(example, tokenizer):
    """
        Tokenizes input examples for a sequence-to-sequence model.

        Parameters
        ----------
        example: dict
            A dictionary containing the input text and target text.
            Expected keys are 'text' for the input sequences and 'target'
            for the target sequences.

        tokenizer: tokenizer
            A tokenizer for a model

        Return
        ------
        inputs : dict
            A dictionary with tokenized inputs and labels. The keys are:
                - 'input_ids': Tokenized input sequence tensors.
                - 'attention_mask': Attention masks for the input sequences.
                - 'labels': Tokenized target sequence tensors.
    """
    # Tokenize the input text with padding and truncation to max_length
    inputs = tokenizer(
        example['text'],
        padding=PADDING_TYPE,
        truncation=TRUNCATION,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=RETURN_TENSORS,
    )
    # Clone input_ids as labels
    inputs['labels'] = inputs['input_ids'].clone()
    return inputs


def create_chat_template(example):
    """
        Creates a formatted prompt.

        Parameters
        ----------
        example: str
            A string containing the instruction.

        Return
        ------
        prompt: str
            A string representing the formatted prompt.
    """
    instruction = example['instruction']
    output =example['output']
    prompt =f"""{PROMPT_TEMPLATE}

Problem: {instruction}
[/INST]
[GDSCRIPT]
{output}
[/GDSCRIPT]"""
    return prompt


def create_tokenized_dataset(tokenizer):
    """
        Creates tokenized datasets for training and validation.

        Parameters
        ----------
        tokenizer : tokenizer
            A pretrained tokenizer.

        Return
        ------
        data : DatasetDict
            A DatasetDict containing 'train' and 'val' datasets.
    """
    # Get dataset from HuggingFace
    raw_dataset = load_dataset(DATASET_NAME)["train"]

    raw_dataset = raw_dataset.map(lambda x: {'text': f"{create_chat_template(x)}"},
                                  remove_columns=['instruction', 'output'])
    
    # Tokenize the dataset
    tokenized_datasets = raw_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])
    # Split the dataset into train and validation
    df = tokenized_datasets.to_pandas()

    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    data = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "val": Dataset.from_pandas(val_df)
    })
    return data


def save_model_and_tokenizer(trainer, tokenizer):
    """
        Saves the fine-tuned model and tokenizer.

        Parameters
        ----------
        trainer : Trainer
            Trainer object used for fine-tuning.
        tokenizer : tokenizer
            A tokenizer.
    """
    # save model
    trainer.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    # save tokenizer
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")


def create_trainer(model, tokenizer, data):
    """
        Creates a Trainer instance for fine-tuning.

        Parameters
        ----------
        model : AutoModelForCausalLM
            Pre-trained model instance.
        tokenizer : tokenizer
            A tokenizer.
        data : DatasetDict
            DatasetDict containing 'train' and 'val' datasets.

        Return
        ------
        trainer : Trainer
            Trainer instance for fine-tuning.
    """
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy=EVAL_STRATEGY,
        output_dir=OUTPUT_DIR,
    )

    lora_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
    )

    model = get_peft_model(model, lora_config).to("cuda")
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
    )
    return trainer


def main():
    """
       Main function to orchestrate the fine-tuning process.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Add padding token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # create tokenized dataset
    data = create_tokenized_dataset(tokenizer)

    # load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    # create Trainer
    trainer = create_trainer(model, tokenizer, data)

    # Start fine-tuning
    trainer.train()
    save_model_and_tokenizer(trainer, tokenizer)


if __name__ == "__main__":
    main()

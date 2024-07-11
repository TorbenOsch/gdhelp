"""
Description: This script takes a prompt and generates the output from an AI model. it loads
the model and the maximum lengths.

Author: Torben Oschkinat - cgt104590 - Bachelor degree

Usage:
    python testing_prompts.py

Requirements:
    - The tokenizer.pkl, should be present in the same directory as this script.
    - The model.keras, should be present in the same directory as this script.
    - The max_lengths.pkl model, should be present in the same directory as this script.
    - The script uses the Python libraries: os, tensorflow, pickle and numpy.
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model

# Configure TensorFlow to allow memory growth for all detected GPUs
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the tokenizer from a pickle file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the maximum sequence lengths from a pickle file
with open('max_lengths.pkl', 'rb') as f:
    max_lengths = pickle.load(f)
    max_input_len = max_lengths['max_input_len']
    max_target_len = max_lengths['max_target_len']


def masked_loss(label, pred):
    """
    Computes the masked sparse categorical cross-entropy loss.

    This function calculates the sparse categorical cross-entropy loss between the
    true labels and predictions while ignoring the padded values.

    Parameters
    ----------
    label : tf.Tensor
        A tensor of true labels with padding tokens (0) included.

    pred : tf.Tensor
        A tensor of predicted values.

    Returns
    -------
    masked_loss : tf.Tensor
        The masked loss value computed by applying a mask to ignore padding tokens
        and then averaging the remaining loss values.
    """

    # Create a mask to ignore the padding tokens
    mask = tf.cast(label != 0, dtype=pred.dtype)
    # Initialize the sparse categorical cross-entropy loss with no reduction
    spars_cat_cross = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    # Compute the sparse categorical cross-entropy loss for each element
    loss = spars_cat_cross(label, pred)
    # Apply the mask to the loss, then sum and normalize by the number of unmasked elements
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return masked_loss


def masked_accuracy(label, pred):
    """
    Computes the masked accuracy.

    This function calculates the accuracy of the predictions while ignoring
    the padded values. The accuracy is computed by comparing
    the predicted labels to the true labels and masking the padded positions.

    Parameters
    ----------
    label : tf.Tensor
        A tensor of true labels with padding tokens (0) included.

    pred : tf.Tensor
        A tensor of predicted logits.

    Returns
    -------
    masked_accuracy : tf.Tensor
        The masked accuracy value computed by applying a mask to ignore padding
        tokens and then averaging the remaining correct predictions.
    """

    # Create a mask to ignore the padding tokens
    mask = tf.cast(label != 0, dtype=tf.float32)
    # Get the predicted class indices by taking the argmax along the last axis
    pred = tf.argmax(pred, axis=-1, output_type=tf.int32)
    # Ensure the true labels are of type int32 for comparison
    label = tf.cast(label, tf.int32)
    # Calculate the correctness of predictions by comparing to true labels
    correct = tf.cast(tf.equal(pred, label), dtype=tf.float32)
    # Apply the mask to the correctness tensor
    masked_correct = correct * mask
    # Compute the masked accuracy by summing the masked correct values and normalizing
    return tf.reduce_sum(masked_correct) / tf.reduce_sum(mask)


custom_objects = {
    'masked_loss': masked_loss,
    'masked_accuracy': masked_accuracy
}

# Load the model
transformer = load_model('model.keras', custom_objects=custom_objects)


def generate_code(prompt):
    """
    Generates code based on a given prompt using a pre-trained transformer model.

    This function tokenizes the input prompt, generates code token-by-token until
    an end-of-sequence (EOS) token is predicted or the maximum target length is reached,
    and then decodes the generated token sequence back into text.

    Parameters
    ----------
    prompt : str
        The input text prompt for which to generate code.

    Returns
    -------
    generated_code : str
        The generated code as a string.
    """

    eos_token = '<EOS>'
    # Tokenize the prompt
    input_seq = tokenizer.texts_to_sequences([prompt])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    # Generate code
    output_seq = np.zeros((1, max_target_len - 1), dtype=int)
    for i in range(max_target_len - 1):
        # Predict the next token
        predictions = transformer.predict([input_seq, output_seq])
        predicted_token_index = np.argmax(predictions[:, i, :], axis=-1)[0]
        if predicted_token_index == tokenizer.word_index[eos_token]:
            # Cut off everthing after EOS
            output_seq = output_seq[:, :i]
            break
        output_seq[0, i] = predicted_token_index
    print("Tokenized output sequence:", output_seq)
    # Decode
    generated_code = tokenizer.sequences_to_texts(output_seq.tolist())[0]
    return generated_code


def preprocess_text(snippet):
    """
    Preprocesses a code snippet by adding spaces around punctuation and special characters.

    Parameters
    ----------
    snippet : str
        The input code snippet to be preprocessed.

    Returns
    -------
    str
        The preprocessed code snippet with spaces added around each punctuation and special character.
    """

    snippet = snippet.replace('\n', ' \n ')
    snippet = snippet.replace('\t', ' \t ')
    snippet = snippet.replace('.', ' . ')
    snippet = snippet.replace(':', ' : ')
    snippet = snippet.replace('_', ' _ ')
    snippet = snippet.replace('=', ' = ')
    snippet = snippet.replace('+', ' + ')
    snippet = snippet.replace('-', ' - ')
    snippet = snippet.replace('*', ' * ')
    snippet = snippet.replace('/', ' / ')
    snippet = snippet.replace('%', ' % ')
    snippet = snippet.replace('>', ' > ')
    snippet = snippet.replace('<', ' < ')
    snippet = snippet.replace('>=', ' >= ')
    snippet = snippet.replace('<=', ' <= ')
    snippet = snippet.replace('==', ' == ')
    snippet = snippet.replace('!=', ' != ')
    snippet = snippet.replace('&', ' & ')
    snippet = snippet.replace('|', ' | ')
    snippet = snippet.replace('&&', ' && ')
    snippet = snippet.replace('||', ' || ')
    snippet = snippet.replace('!', ' ! ')
    snippet = snippet.replace('?', ' ? ')
    snippet = snippet.replace(',', ' , ')
    snippet = snippet.replace(';', ' ; ')
    snippet = snippet.replace('{', ' { ')
    snippet = snippet.replace('}', ' } ')
    snippet = snippet.replace('[', ' [ ')
    snippet = snippet.replace(']', ' ] ')
    snippet = snippet.replace('(', ' ( ')
    snippet = snippet.replace(')', ' ) ')
    snippet = snippet.replace('"', ' " ')
    snippet = snippet.replace("'", " ' ")
    return snippet


prompt = "Free up the memory used by the current node instance."
generated_code = generate_code(preprocess_text(prompt))


def reverse_preprocess_text(snippet):
    """
    Reverses the preprocessing of a code snippet by removing added spaces around punctuation and special characters.

    Parameters
    ----------
    snippet : str
        The preprocessed code snippet to be reversed.

    Returns
    -------
    str
        The code snippet with spaces around punctuation and special characters removed.
    """
    snippet = snippet.replace(' \n ', '\n')
    snippet = snippet.replace(' \t ', '\t')
    snippet = snippet.replace(' . ', '.')
    snippet = snippet.replace(' : ', ':')
    snippet = snippet.replace(' _ ', '_')
    snippet = snippet.replace(' = ', '=')
    snippet = snippet.replace(' + ', '+')
    snippet = snippet.replace(' - ', '-')
    snippet = snippet.replace(' * ', '*')
    snippet = snippet.replace(' / ', '/')
    snippet = snippet.replace(' % ', '%')
    snippet = snippet.replace(' > ', '>')
    snippet = snippet.replace(' < ', '<')
    snippet = snippet.replace(' >= ', '>=')
    snippet = snippet.replace(' <= ', '<=')
    snippet = snippet.replace(' == ', '==')
    snippet = snippet.replace(' != ', '!=')
    snippet = snippet.replace(' & ', '&')
    snippet = snippet.replace(' | ', '|')
    snippet = snippet.replace(' && ', '&&')
    snippet = snippet.replace(' || ', '||')
    snippet = snippet.replace(' ! ', '!')
    snippet = snippet.replace(' ? ', '?')
    snippet = snippet.replace(' , ', ',')
    snippet = snippet.replace(' ; ', ';')
    snippet = snippet.replace(' { ', '{')
    snippet = snippet.replace(' } ', '}')
    snippet = snippet.replace(' [ ', '[')
    snippet = snippet.replace(' ] ', ']')
    snippet = snippet.replace(' ( ', '(')
    snippet = snippet.replace(' ) ', ')')
    snippet = snippet.replace(' " ', '"')
    snippet = snippet.replace(" ' ", "'")
    return snippet


generated_code = reverse_preprocess_text(generated_code)
print("Generated Code:")
print(generated_code)

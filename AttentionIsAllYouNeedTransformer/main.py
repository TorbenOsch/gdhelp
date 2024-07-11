"""
Description: This script creates a tranformer like the one in Attention is all you need.
It trains it on a given dataset in a csv file and sves the trained tranformer and tokenizer.

Author: Torben Oschkinat - cgt104590 - Bachelor degree

Usage:
    python main.py

Requirements:
    - The csv file, should be present in the same directory as this script.
    - The script uses the Python libraries: os, tensorflow, pandas, pickle and numpy.
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Constants - Hyperparameter
EPOCHS = 1
PATIENCE = 3
BATCH_SIZE = 1
HEAD_SIZE = 4
NUM_HEADS = 8
FF_DIM = 16
NUM_TRANSFORMER_BLOCKS = 1
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001
SPLIT_RATIO = 0.2

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

dataset_name = 'data.csv'
df = pd.read_csv(dataset_name, sep=';', encoding='latin1')

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


# Preprocess the 'gdscript_code' column
df['gdscript_code'] = df['gdscript_code'].apply(preprocess_text)
# Preprocess the 'prompt' column
df['prompt'] = df['prompt'].apply(preprocess_text)

# Append the end-of-sequence token
eos_token = '<EOS>'
df['gdscript_code'] = df['gdscript_code'].apply(lambda x: x + ' ' + eos_token)

# Combine all texts from 'prompt' and 'gdscript_code' columns into a single list.
all_texts = df['prompt'].tolist() + df['gdscript_code'].tolist()

# Fit the Tokenizer on all_texts
tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(all_texts)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert text and code sequences
input_sequences = tokenizer.texts_to_sequences(df['prompt'].tolist())
target_sequences = tokenizer.texts_to_sequences(df['gdscript_code'].tolist())

# Define input and target sequence lengths
vocab_size = len(tokenizer.word_index) + 1
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)
input_length = max_input_len
target_length = max_target_len - 1

with open('max_lengths.pkl', 'wb') as f:
    pickle.dump({'max_input_len': max_input_len, 'max_target_len': max_target_len}, f)

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Split target sequences into training and validation sets
indices = np.arange(input_sequences.shape[0])
np.random.shuffle(indices)
split_index = int((1 - SPLIT_RATIO) * input_sequences.shape[0])
train_indices = indices[:split_index]
val_indices = indices[split_index:]
X_train = input_sequences[train_indices]
y_train = target_sequences[train_indices]
X_val = input_sequences[val_indices]
y_val = target_sequences[val_indices]


def create_dataset(inputs, targets, batch_size):
    """
    Creates a TensorFlow Dataset from input and target tensors.

    This function constructs a TensorFlow Dataset object from input and target tensors,
    shuffles it, batches it, and prefetches batches for improved performance.

    Parameters
    ----------
    inputs : tf.Tensor
        Tensor containing input sequences.

    targets : tf.Tensor
        Tensor containing target sequences corresponding to the input sequences.

    batch_size : int
        Batch size for training or evaluation.

    Returns
    -------
    tf.data.Dataset
        A TensorFlow Dataset
    """

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(buffer_size=len(inputs))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_dataset((X_train, y_train[:, :-1]), y_train[:, 1:], BATCH_SIZE)
val_dataset = create_dataset((X_val, y_val[:, :-1]), y_val[:, 1:], BATCH_SIZE)


def get_positional_encoding(length, depth):
    """
    Generates positional encodings for input sequences.

    This function calculates positional encodings as described in the Transformer
    architecture.

    Parameters
    ----------
    length : int
        Length of the input sequence.

    depth : int
        Depth of the positional encoding, which determines the number of dimensions.

    Returns
    -------
    tf.Tensor
        Positional encoding tensor of shape (length, depth).
    """

    # Create arrays for positions and depths
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth // 2)[np.newaxis, :] / (depth // 2)
    # Calculate angle radians based on positions and depths
    angle_rads = positions / (10000**depths)
    # Compute sine and cosine of angle radians and concatenate them
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def transformer_encoder(inputs, zero_mask, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformer encoder block.

    This function implements a single encoder block of the Transformer architecture,
    including multi-head self-attention and feed-forward layers.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor for the encoder block.

    zero_mask : tf.Tensor
        Mask tensor for masking padding tokens in the input sequence.

    head_size : int
        Dimensionality of each attention head.

    num_heads : int
        Number of attention heads in each multi-head attention layer.

    ff_dim : int
        Dimensionality of the feed-forward layer.

    dropout : float, optional, default=0.1
        Dropout rate for regularization.

    Returns
    -------
    tf.Tensor
        Output tensor

    tf.Tensor
        Mask tensor used for masking padding tokens in the input sequence.
    """

    # Multi-Head Self-Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, value=inputs, attention_mask=zero_mask)
    # Add & Norm
    x = layers.Add()([inputs, x])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Residual connection
    res_x = x
    # Feed-forward Network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    # Add & Norm
    x = layers.Add()([res_x, x])
    return layers.LayerNormalization(epsilon=1e-6)(x), zero_mask


def transformer_decoder(inputs, enc_outputs, dec_zero_mask, enc_zero_mask, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformer decoder block.

    This function implements a single decoder block of the Transformer architecture,
    including masked multi-head self-attention, attention with encoder outputs, and
    feed-forward layers.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor for the decoder block.

    enc_outputs : tf.Tensor
        Encoder outputs used for attention mechanisms.

    dec_zero_mask : tf.Tensor
        Mask tensor for masking future tokens in the decoder input sequence.

    enc_zero_mask : tf.Tensor
        Mask tensor for masking padding tokens in the encoder output sequence.

    head_size : int
        Dimensionality of each attention head.

    num_heads : int
        Number of attention heads in each multi-head attention layer.

    ff_dim : int
        Dimensionality of the feed-forward layer.

    dropout : float, optional, default=0.1
        Dropout rate for regularization.

    Returns
    -------
    tf.Tensor
        Output tensor

    tf.Tensor
        Mask tensor used for masking future tokens in the decoder input sequence.
    """

    # Masked Multi-Head Self-Attention
    input_shape = tf.shape(inputs)
    seq_length = input_shape[1]
    causal_mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    combined_mask = tf.minimum(dec_zero_mask, causal_mask)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, value=inputs, attention_mask=combined_mask)
    # Add & Norm:
    x = layers.Add()([inputs, x])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Residual connection
    res_x = x
    # Multi-Head Attention with Encoder Outputs
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(query=x, key=enc_outputs, value=enc_outputs, attention_mask=enc_zero_mask)
    # Add & Norm
    x = layers.Add()([res_x, x])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Residual connection
    res_x = x
    # Feed-forward Network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    # Add & Norm
    x = layers.Add()([res_x, x])
    return layers.LayerNormalization(epsilon=1e-6)(x), dec_zero_mask

def build_transformer(vocab_size, input_length, target_length, head_size, num_heads, ff_dim, num_transformer_blocks, dropout=0.1):
    """
    Builds a Transformer model.

    This function constructs a Transformer model composed of encoder and decoder blocks.
    The encoder processes input sequences, while the decoder predicts output sequences
    conditioned on the encoder's output.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary, determines the embedding matrix dimensions.

    input_length : int
        Length of the input sequence.

    target_length : int
        Length of the target sequence.

    head_size : int
        Dimensionality of each attention head.

    num_heads : int
        Number of attention heads in each multi-head attention layer.

    ff_dim : int
        Dimensionality of the feed-forward layer.

    num_transformer_blocks : int
        Number of transformer blocks (encoder and decoder) to stack.

    dropout : float, optional, default=0.1
        Dropout rate for regularization.

    Returns
    -------
    keras.Model
        A Transformer model
    """

    # Define encoder inputs for the input sequence
    inputs = layers.Input(shape=(input_length,))
    # Embedding layer for the encoder
    enc_embedding = layers.Embedding(vocab_size, head_size,)(inputs)
    # Get positional encoding for the input sequence
    enc_positional_encoding = get_positional_encoding(input_length, head_size)
    # Add positional encoding to the encoder embeddings
    enc_outputs = enc_embedding + enc_positional_encoding
    # Create a mask to ignore padding tokens in the input sequence
    enc_zero_mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
    # Build the encoder with multiple transformer blocks
    for _ in range(num_transformer_blocks):
        enc_outputs, enc_mask = transformer_encoder(enc_outputs, enc_zero_mask, head_size, num_heads, ff_dim, dropout)
    # Define decoder inputs for the target sequence
    dec_inputs = layers.Input(shape=(target_length,))
    # Embedding layer for the decoder
    dec_embedding = layers.Embedding(vocab_size, head_size)(dec_inputs)
    # Get positional encoding for the target sequence
    dec_positional_encoding = get_positional_encoding(target_length, head_size)
    # Add positional encoding to the decoder embeddings
    dec_outputs = dec_embedding + dec_positional_encoding
    # Create a mask to ignore padding tokens in the target sequence
    dec_zero_mask = tf.cast(tf.not_equal(dec_inputs, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
    # Build the decoder with multiple transformer blocks
    for _ in range(num_transformer_blocks):
        dec_outputs, dec_mask = transformer_decoder(dec_outputs, enc_outputs, dec_zero_mask, enc_zero_mask, head_size, num_heads, ff_dim, dropout)
    # Output layer for predicting probabilities over the vocabulary
    outputs = layers.Dense(vocab_size, activation="softmax")(dec_outputs)
    return Model([inputs, dec_inputs], outputs)


transformer = build_transformer(vocab_size, input_length, target_length, HEAD_SIZE, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS, DROPOUT_RATE)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

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


transformer.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])
transformer.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

transformer.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[early_stopping])

transformer.save('model.keras')

import numpy as np
import tensorflow as tf


class InvalidInputException(Exception):
    def __init__(self, n_dims):
        self.message = f"Invalid input vector: 3-dimensional expected, transferred {n_dims} dimension(-s)"
        super().__init__(self.message)


class CellGRU(tf.Module):
    """
    :param units - units (number of neurons) that used to compute the hidden state.
    """

    def __init__(self, units: int, input_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)

        # Use Glorot Uniform initializer for weights initializing, zeros initializer for bias.
        w_initializer = tf.initializers.GlorotUniform(seed=seed)
        zeros_initializer = tf.initializers.Zeros()

        # Initialize reset gate weights.
        self.W_x_r = tf.Variable(w_initializer(shape=(input_length, units)), dtype=tf.float32,
                                 name=f"W_x_r_{self.name}", trainable=True)

        self.W_h_r = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_r_{self.name}", trainable=True)

        self.b_r = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_r_{self.name}", trainable=use_bias)

        # Initialize update gate weights.
        self.W_x_u = tf.Variable(w_initializer(shape=(input_length, units)), dtype=tf.float32,
                                 name=f"W_x_u_{self.name}", trainable=True)

        self.W_h_u = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_u_{self.name}", trainable=True)

        self.b_u = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_u_{self.name}", trainable=use_bias)

        # Initialize candidate cell state weights.
        self.W_x_h = tf.Variable(w_initializer(shape=(input_length, units)), dtype=tf.float32,
                                 name=f"W_x_h_{self.name}", trainable=True)

        self.W_h_h = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_h_{self.name}", trainable=True)

    def __call__(self, x: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
        r = tf.sigmoid(tf.matmul(x, self.W_x_u) + tf.matmul(h, self.W_h_u) + self.b_r)  # what should be remembered
        u = tf.sigmoid(tf.matmul(x, self.W_x_u) + tf.matmul(h, self.W_h_u) + self.b_u)  # what should be updated

        h_cand = tf.tanh(tf.matmul(x, self.W_x_h) + tf.matmul(r * h, self.W_h_h))  # what we can add
        h_new = (1 - u) * h_cand + u * h  # what we got after adding (h_t)

        return h_new


class GRU(tf.Module):
    def __init__(self, units: int, input_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)
        self.units = units
        self.cell = CellGRU(units, input_length, use_bias, seed)

    def __call__(self, x):
        batch_size, sequence_length, input_length = tf.shape(x)

        # Initialize h_0
        h = tf.zeros([batch_size, self.units])
        hidden_states = []

        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)
            hidden_states.append(h)

        # Transform list to tensor.
        hidden_states = tf.stack(hidden_states, axis=1)  # (batch_size, seq_len, hidden_dim)
        return hidden_states


class TranslatorRNN(tf.Module):
    """

    """
    def __init__(self, encoder_units: int, decoder_units: int, features_number: int, learning_rate=0.001,
                 regularization_strength=0., use_bias=True, seed=None, name=None):
        super().__init__(name=name)

        # Initialize encoder layer.
        self.encoder = GRU(encoder_units, features_number, use_bias=use_bias, seed=seed)

    def __call__(self, x):
        # Input Data should be 3-dimensional.
        if x.shape.ndims != 3:
            raise InvalidInputException(x.shape.ndims)

        return self.encoder(x)

    def __str__(self):
        # return f"Simple Translator RNN\n\tEncoder: {self.encoder}\n\tDecoder: {self.decoder}"
        return f"Simple Translator RNN\n\tEncoder: {self.encoder}"

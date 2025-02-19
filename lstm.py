# Copyright [2024] [Nikita Karpov]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf


class CellLSTM(tf.Module):
    """
    :param units - units (number of neurons) that used to compute the hidden state.
    :param token_length - number of x's features or length of one token (symbol/word/etc).
    """
    def __init__(self, units: int, token_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)

        # Use Glorot Uniform initializer for weights initializing, zeros initializer for bias.
        w_initializer = tf.initializers.GlorotUniform(seed=seed)
        zeros_initializer = tf.initializers.Zeros()

        # Initialize forget gate weights. Use union matrix for stacked x and h.
        self.W_f = tf.Variable(w_initializer(shape=(token_length + units, units)), dtype=tf.float32,
                                 name=f"W_f_{self.name}", trainable=True)

        self.b_f = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_f_{self.name}", trainable=use_bias)

        # Initialize input gate weights. Use union matrix too.
        self.W_i = tf.Variable(w_initializer(shape=(token_length + units, units)), dtype=tf.float32,
                                 name=f"W_i_{self.name}", trainable=True)

        self.b_i = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_i_{self.name}", trainable=use_bias)

        # Initialize cell candidate gate weights. Use union matrix.
        self.W_g = tf.Variable(w_initializer(shape=(token_length + units, units)), dtype=tf.float32,
                                 name=f"W_g_{self.name}", trainable=True)

        self.b_g = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_g_{self.name}", trainable=use_bias)

        # Initialize output gate weights. Use union matrix too.
        self.W_o = tf.Variable(w_initializer(shape=(token_length + units, units)), dtype=tf.float32,
                                 name=f"W_o_{self.name}", trainable=True)

        self.b_o = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_o_{self.name}", trainable=use_bias)


    def __call__(self, x: tf.Tensor, h: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        x_h = tf.concat([x, h], axis=-1)

        i = tf.sigmoid(tf.matmul(x_h, self.W_i) + self.b_i)  # what should be remembered
        f = tf.sigmoid(tf.matmul(x_h, self.W_f) + self.b_f)  # what should be forgotten
        o = tf.sigmoid(tf.matmul(x_h, self.W_o) + self.b_o)  # what should be added to h
        g = tf.tanh(tf.matmul(x_h, self.W_g) + self.b_g)     # new context (candidate cell) 

        c_new = f * c + i * g       # update new context
        h_new = o * tf.tanh(c_new)  # count new hiddent state

        return h_new, c_new
    

class DecoderLSTM(tf.Module):
    """
    Decoder based on LSTM that used to predict translation from context and hidden state by encoder.
    """
    def __init__(self, input_length: int, output_token_length: int, max_sequence_size: int, use_bias: bool, seed: int, name=None):
        super().__init__(name)
        self.max_sequence_size = max_sequence_size
        self.output_token_length = output_token_length
        
        # Decoder LSTM.
        self.lstm = CellLSTM(units=input_length, token_length=output_token_length, use_bias=use_bias, seed=seed)

        # Weights for y prediction.
        self.W_y = tf.Variable(tf.initializers.GlorotUniform(seed=seed)(shape=(input_length, output_token_length)), dtype=tf.float32)
        self.b_y = tf.Variable(tf.zeros(shape=(output_token_length,)), dtype=tf.float32)


    def __call__(self, s_0: tf.Tensor, c_0: tf.Tensor):
        batch_size = s_0.shape[0]

        START_TOKEN = tf.zeros(shape=(batch_size, self.output_token_length), dtype=tf.float32)
        STOP_TOKEN = tf.ones(shape=(batch_size, self.output_token_length), dtype=tf.float32)
        
        # Current output of model.
        current_sequence_size = 0

        # List with predicted translation.
        y_predicts = [START_TOKEN]
        
        y = y_predicts[0]
        s = s_0  # decoder hidden states
        c = c_0  # decoder context

        # Prediction works while max output sequence len not reached or while LSTM predicts STOP token.
        while current_sequence_size < self.max_sequence_size and tf.reduce_any(tf.not_equal(y, STOP_TOKEN)):
            s, c = self.lstm(y_predicts[-1], s, c)

            # Predict current word y (embedded) from new hidden state s 
            y = tf.sigmoid(tf.matmul(s, self.W_y) + self.b_y)
            y_predicts.append(y)

            current_sequence_size += 1

        return tf.stack(y_predicts, axis=1)

    def __str__(self):
        return f"Decoder based on LSTM"
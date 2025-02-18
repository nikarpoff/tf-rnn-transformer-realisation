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
        self.W_f = tf.Variable(w_initializer(shape=(token_length, units * 2)), dtype=tf.float32,
                                 name=f"W_f_{self.name}", trainable=True)

        self.b_f = tf.Variable(zeros_initializer(shape=(units * 2,)), name=f"b_f_{self.name}", trainable=use_bias)

        # Initialize input gate weights. Use union matrix too.
        self.W_i = tf.Variable(w_initializer(shape=(token_length, units * 2)), dtype=tf.float32,
                                 name=f"W_i_{self.name}", trainable=True)

        self.b_i = tf.Variable(zeros_initializer(shape=(units * 2,)), name=f"b_i_{self.name}", trainable=use_bias)

        # Initialize cell candidate gate weights. Use union matrix.
        self.W_g = tf.Variable(w_initializer(shape=(token_length, units * 2)), dtype=tf.float32,
                                 name=f"W_g_{self.name}", trainable=True)

        self.b_g = tf.Variable(zeros_initializer(shape=(units * 2,)), name=f"b_g_{self.name}", trainable=use_bias)

        # Initialize output gate weights. Use union matrix too.
        self.W_o = tf.Variable(w_initializer(shape=(token_length, units * 2)), dtype=tf.float32,
                                 name=f"W_o_{self.name}", trainable=True)

        self.b_o = tf.Variable(zeros_initializer(shape=(units * 2,)), name=f"b_o_{self.name}", trainable=use_bias)


    def __call__(self, x: tf.Tensor, h: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        x_h = tf.stack(x, h, axis=0)

        i = tf.sigmoid(tf.matmul(x_h, self.W_i) + self.b_i)  # what should be remembered
        f = tf.sigmoid(tf.matmul(x_h, self.W_f) + self.b_f)  # what should be forgotten
        o = tf.sigmoid(tf.matmul(x_h, self.W_o) + self.b_o)  # what should be added to h
        g = tf.tanh(tf.matmul(x_h, self.W_g) + self.b_g)     # new context (candidate cell) 

        c_new = tf.matmul(f, c) + tf.matmul(i, g)  # update new context
        h_new = tf.matmul(o, tf.tanh(c_new))       # count new hiddent state

        return h_new, c_new
    

class DecoderLSTM(tf.Module):
    def __init__(self, units: int, token_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name)

        self.lstm = CellLSTM(units, token_length, use_bias, seed, f"{name}-CellLSTM")

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

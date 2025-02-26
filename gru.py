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


class CellGRU(tf.Module):
    """
    :param units - units (number of neurons) that used to compute the hidden state.
    :param token_length - number of x's features or length of one token (symbol/word/etc).
    """

    def __init__(self, units: int, token_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)

        # Use Glorot Uniform initializer for weights initializing, zeros initializer for bias.
        w_initializer = tf.initializers.GlorotUniform(seed=seed)
        zeros_initializer = tf.initializers.Zeros()

        # Initialize reset gate weights.
        self.W_x_r = tf.Variable(w_initializer(shape=(token_length, units)), dtype=tf.float32,
                                 name=f"W_x_r_{self.name}", trainable=True)

        self.W_h_r = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_r_{self.name}", trainable=True)

        self.b_r = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_r_{self.name}", trainable=use_bias)

        # Initialize update gate weights.
        self.W_x_u = tf.Variable(w_initializer(shape=(token_length, units)), dtype=tf.float32,
                                 name=f"W_x_u_{self.name}", trainable=True)

        self.W_h_u = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_u_{self.name}", trainable=True)

        self.b_u = tf.Variable(zeros_initializer(shape=(units,)), name=f"b_u_{self.name}", trainable=use_bias)

        # Initialize candidate cell state weights.
        self.W_x_h = tf.Variable(w_initializer(shape=(token_length, units)), dtype=tf.float32,
                                 name=f"W_x_h_{self.name}", trainable=True)

        self.W_h_h = tf.Variable(w_initializer(shape=(units, units)), dtype=tf.float32,
                                 name=f"W_h_h_{self.name}", trainable=True)

    def __call__(self, x: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
        r = tf.sigmoid(tf.matmul(x, self.W_x_r) + tf.matmul(h, self.W_h_r) + self.b_r)  # what should be remembered
        u = tf.sigmoid(tf.matmul(x, self.W_x_u) + tf.matmul(h, self.W_h_u) + self.b_u)  # what should be updated

        h_cand = tf.tanh(tf.matmul(x, self.W_x_h) + tf.matmul(r * h, self.W_h_h))  # what we can add
        h_new = (1 - u) * h_cand + u * h  # what we got after adding (h_t)

        return h_new
    
class GRU(tf.Module):
    def __init__(self, units: int, token_length: int, use_bias: bool, return_sequences: bool, seed: int, name=None):
        super().__init__(name=name)
        self.units = units
        self.token_length = token_length
        self.return_sequences = return_sequences
        self.cell = CellGRU(units, token_length, use_bias, seed)

    def __call__(self, x):
        batch_size, sequence_length, _ = tf.shape(x)

        # Initialize h_0.
        h = tf.zeros([batch_size, self.units])
        hidden_states = []  # list of (batch_size, units) outputs with sequence_length size
        
        # Every element of sequence is x on one step.
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)
            hidden_states.append(h)

        if self.return_sequences:
            # Transform list to tensor (stacked by batches).
            hidden_states = tf.stack(hidden_states, axis=1)  # (batch_size, seq_len, hidden_dim)
            return hidden_states
        else:
            return h

    def __str__(self):
        if self.return_sequences:
            output_shape = f"(None, sequence_length, {self.units})"
        else:
            output_shape = f"(None, {self.units})"

        return f"GRU layer. Output shape - {output_shape}"

class DeepEncoderGRU(tf.Module):
    """
    Realisation of multilayered RNN based on GRU.

    Last layer returns context of sequence. It always have shape (batch_size, token_length * 2).
    """
    def __init__(self, units_list: list, token_length: int, use_bias=True, seed=None, name=None):
        super().__init__(name)
        
        # First layer has input shape (batch_size, seq_len, token_length).
        self.deep_model = [
            GRU(
                units=units_list[0],
                token_length=token_length,
                use_bias=use_bias,
                return_sequences=True,
                seed=seed,
            ),
        ]

        # Another user specified layers have input shape (batch_size, seq_len, units_prev_layer).
        for i in range(1, len(units_list) - 1):
            self.deep_model.append(GRU(
                units=units_list[i],
                token_length=units_list[i - 1],
                use_bias=use_bias,
                return_sequences=True,
                seed=seed,
            ))

        # Last layer can not return sequences.
        self.deep_model.append(GRU(
            units=units_list[-1],
            token_length=units_list[-2],
            use_bias=use_bias,
            return_sequences=False,
            seed=seed,
        ))

    def __call__(self, x):
        h = x

        for layer in self.deep_model:
            h = layer(h)

        return h
    
    def __str__(self):
        model_str = f"Multilayer RNN with {len(self.deep_model)} layers:"

        for sub_model in self.deep_model:
            model_str += f"\n\t\t\t{sub_model}"

        return model_str

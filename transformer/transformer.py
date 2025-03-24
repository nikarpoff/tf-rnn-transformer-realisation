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
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, token_length, max_len, name=None):
        super().__init__(name=name)
        self.token_length = token_length
        self.max_len = max_len

        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]

        # sinusoidal absolute, wk = 1 / 1000 ^ (2k / d)
        omega = tf.exp(
            tf.range(0, token_length, 2, dtype=tf.float32) * 
            (-np.log(10000.0) / token_length)
        )

        sin_part = tf.sin(positions * omega)
        cos_part = tf.cos(positions * omega)

        self.pos_encoding = tf.reshape(
            tf.stack([sin_part, cos_part], axis=-1),
            (max_len, token_length)
        )

        self.pos_encoding = self.pos_encoding[tf.newaxis, ...]  # add batch dimension
        
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    
    def get_config(self):
        return {
            "token_length": self.token_length,
            "max_len": self.max_len,
            "name": self.name,
        }

class MultiHeadAttentionUnit(tf.keras.layers.Layer):
    def __init__(self, d_k: int, token_length: int, heads_number: int, seed: int, name=None):
        super().__init__(name=name)
        self.d_k = d_k
        self.seed = seed

        self.heads_number = heads_number
        self.token_length = token_length

        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        # Use Glorot Uniform initializer for weights initializing.
        w_initializer = tf.initializers.GlorotUniform(seed=self.seed)

        # Initialize query matrix weigths. Use MultiHead matrix.
        self.W_q = self.add_weight(shape=(self.token_length, self.d_k * self.heads_number), initializer=w_initializer, name=f"W_q")
        
        # Initialize key matrix weigths. Use MultiHead matrix.
        self.W_k = self.add_weight(shape=(self.token_length, self.d_k * self.heads_number), initializer=w_initializer, name=f"W_k")

        # Initialize value matrix weigths. Use MultiHead matrix.
        self.W_v = self.add_weight(shape=(self.token_length, self.d_k * self.heads_number), initializer=w_initializer, name=f"W_v")

        # Initialize output matrix weights.
        self.W_o = self.add_weight(shape=(self.d_k * self.heads_number, self.token_length), name=f"W_o")

        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.heads_number, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x: tf.Tensor, y: tf.Tensor, mask=None):
        batch_size = tf.shape(x)[0]

        # Compute Q, K, V projections.
        Q = tf.matmul(x, self.W_q)  # (batch, x_len, d_k * heads_number)
        V = tf.matmul(y, self.W_v)  # (batch, y_len, d_k * heads_number)
        K = tf.matmul(y, self.W_k)  # (batch, y_len, d_k * heads_number)

        # Split to heads.
        Q = self.split_heads(Q, batch_size)
        V = self.split_heads(V, batch_size)
        K = self.split_heads(K, batch_size)

        # Compute attention logits E = QK^T / sqrt(depth).
        E = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))

        # Mask logits if required.
        if mask is not None:
            E += (mask * -1e9)

        # Compute attention matrix along the K axis (y_len).
        A = self.softmax(E)

        # Compute results for every head (multihead output).
        M_O = tf.matmul(A, V)

        # Concatenate results from heads.
        M_O = tf.transpose(M_O, perm=[0, 2, 1, 3])
        M_O = tf.reshape(M_O, (batch_size, -1, self.d_k * self.heads_number))

        # Compute output.
        O = tf.matmul(M_O, self.W_o)

        return O

    def get_config(self):
        return {
            "d_k": self.d_k,
            "token_length": self.token_length,
            "heads_number": self.heads_number,
            "seed": self.seed,
            "name": self.name,
        }

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, model_depth: int, mlp_units: int, token_length: int, max_sequence_length: int, heads_number: int, seed=None, name=None):
        super().__init__(name=name)
        self.model_depth = model_depth
        self.mlp_units = mlp_units
        self.token_length = token_length
        self.max_sequence_length = max_sequence_length
        self.heads_number = heads_number
        self.seed = seed

        self.positional_encoding = PositionalEncoding(token_length, max_sequence_length)
        self.mha = MultiHeadAttentionUnit(model_depth, token_length, heads_number, seed=seed)
        
        self.layer_normalization_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_units, activation="gelu"),
            tf.keras.layers.Dense(token_length)
        ])

        self.layer_normalization_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None):
        h = self.positional_encoding(x)  # positional encoding
        a = self.mha(h, h, mask)  # attention

        h_a = h + a  # residual connection
        h_a_n = self.layer_normalization_1(h_a)  # layer normalization

        mlp_o = self.mlp(h_a_n)  # feed forward

        output = mlp_o + h_a_n  # residual connection
        output_n = self.layer_normalization_2(output)  # layer normalization

        return output_n

    def get_config(self):
        return {
            "max_sequence_length": self.max_sequence_length,
            "heads_number": self.heads_number,
            "token_length": self.token_length,
            "model_depth": self.model_depth,
            "mlp_units": self.mlp_units,
            "seed": self.seed,
            "name": self.name,
        }

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, model_depth: int, mlp_units: int, token_length: int, max_sequence_length: int, heads_number: int, seed=None, name=None):
        super().__init__(name=name)
        self.model_depth = model_depth
        self.mlp_units = mlp_units
        self.token_length = token_length
        self.max_sequence_length = max_sequence_length
        self.heads_number = heads_number
        self.seed = seed

        self.positional_encoding = PositionalEncoding(token_length, max_sequence_length)
        
        self.masked_mha = MultiHeadAttentionUnit(model_depth, token_length, heads_number, seed=seed)
        self.cross_mha = MultiHeadAttentionUnit(model_depth, token_length, heads_number, seed=seed)


        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_units, activation="gelu"),
            tf.keras.layers.Dense(token_length)
        ])

        self.layer_normalization_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalization_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, y, encoder_output, look_ahead_mask, padding_mask=None):
        h = self.positional_encoding(y)  # positional encoding
        a1 = self.masked_mha(h, h, look_ahead_mask)  # masked self attention

        h_a1 = h + a1  # residual connection
        h_a1_n = self.layer_normalization_1(h_a1)  # layer normalization

        a2 = self.cross_mha(h_a1_n, encoder_output, padding_mask)  # cross attention
        h_a1_a2 = h_a1_n + a2
        h_a1_a2_n = self.layer_normalization_2(h_a1_a2)  # layer normalization

        mlp_o = self.mlp(h_a1_a2_n)  # feed forward

        output = mlp_o + h_a1_a2_n  # residual connection
        output_n = self.layer_normalization_3(output)  # layer normalization

        return output_n
    
    def get_config(self):
        return {
            "max_sequence_length": self.max_sequence_length,
            "heads_number": self.heads_number,
            "token_length": self.token_length,
            "model_depth": self.model_depth,
            "mlp_units": self.mlp_units,
            "seed": self.seed,
            "name": self.name,
        }
    
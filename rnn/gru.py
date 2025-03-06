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


class CellGRU(tf.keras.layers.Layer):
    """
    :param units - units (number of neurons) that used to compute the hidden state.
    :param token_length - number of x's features or length of one token (symbol/word/etc).
    """

    def __init__(self, units: int, token_length: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)
        self.units = units
        self.token_length = token_length
        self.use_bias = use_bias
        self.seed = seed

    def build(self, input_shape):
        # Use Glorot Uniform initializer for weights initializing, zeros initializer for bias.
        w_initializer = tf.initializers.GlorotUniform(seed=self.seed)

        # Initialize reset gate weights.
        self.W_x_r = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_x_r")
        self.W_h_r = self.add_weight(shape=(self.units, self.units), initializer=w_initializer, name=f"W_h_r")

        # Initialize update gate weights.
        self.W_x_u = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_x_u")
        self.W_h_u = self.add_weight(shape=(self.units, self.units), initializer=w_initializer, name=f"W_h_u")

        # Initialize candidate cell state weights.
        self.W_x_h = self.add_weight(shape=(self.token_length, self.units), initializer=w_initializer, name=f"W_x_h")
        self.W_h_h = self.add_weight(shape=(self.units, self.units), initializer=w_initializer, name=f"W_h_h")

        if self.use_bias:
            self.b_r = self.add_weight(shape=(self.units,), initializer='zeros', name=f"b_r")
            self.b_u = self.add_weight(shape=(self.units,), initializer='zeros', name=f"b_u")

        super().build(input_shape)

    def call(self, x: tf.Tensor, h: tf.Tensor):
        r = tf.sigmoid(tf.matmul(x, self.W_x_r) + tf.matmul(h, self.W_h_r) + (self.b_r if self.use_bias else 0))  # what should be remembered
        u = tf.sigmoid(tf.matmul(x, self.W_x_u) + tf.matmul(h, self.W_h_u) + (self.b_u if self.use_bias else 0))  # what should be updated

        h_cand = tf.tanh(tf.matmul(x, self.W_x_h) + tf.matmul(r * h, self.W_h_h))  # what we can add
        h_new = (1 - u) * h_cand + u * h  # what we got after adding (h_t)

        return h_new
    
    def get_config(self):
        return {
            "units": self.units,
            "token_length": self.token_length,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }


class EncoderGRU(tf.keras.layers.Layer):
    def __init__(self, units: int, token_length: int, use_bias: bool, return_sequences: bool, seed: int, name=None):
        super().__init__(name=name)
        self.units = units
        self.token_length = token_length
        self.return_sequences = return_sequences
        self.use_bias = use_bias
        self.seed = seed
        self.cell = CellGRU(units, token_length, use_bias, seed)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]

        # The one step of GRU computation.
        def loop_body(step, h, hidden_states):
            x_t = x[:, step, :]
            h_next = self.cell(x_t, h)  # compute new h
            h_next.set_shape([None, self.units])  # h shape is constant
            hidden_states = hidden_states.write(step, h_next)  # store h at current step
            return step + 1, h_next, hidden_states

        # Initialize h_0 and array for all of h.
        h = tf.zeros(shape=(batch_size, self.units), dtype=tf.float32)
        hidden_states = tf.TensorArray(dtype=tf.float32, size=time_steps)

        # Run loop with shape invariants
        _, h, hidden_states = tf.while_loop(
            cond=lambda step, *_: step < time_steps,
            body=loop_body,
            loop_vars=(0, h, hidden_states)
        )

        if self.return_sequences:
            # Stack hidden states to (seq_len, batch_size, units) and transpose
            return tf.transpose(hidden_states.stack(), [1, 0, 2])
        return h

    def __str__(self):
        if self.return_sequences:
            output_shape = f"(None, sequence_length, {self.units})"
        else:
            output_shape = f"(None, {self.units})"

        return f"GRU layer. Output shape - {output_shape}"
    
    def get_config(self):
        return {
            "units": self.units,
            "token_length": self.token_length,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }


class DeepEncoderGRU(tf.keras.layers.Layer):
    """
    Realisation of multilayered RNN based on GRU.

    Last layer returns context of sequence. It always have shape (batch_size, token_length * 2).
    """
    def __init__(self, units_list: list, token_length: int, use_bias=True, return_sequences=False, seed=None, name=None):
        super().__init__(name=name)
        self.units_list = units_list
        self.token_length = token_length
        self.use_bias = use_bias
        self.seed = seed
        
        # First layer has input shape (batch_size, seq_len, token_length).
        self.deep_model = [
            EncoderGRU(
                units=units_list[0],
                token_length=token_length,
                use_bias=use_bias,
                return_sequences=True,
                seed=seed,
            ),
        ]

        # Another user specified layers have input shape (batch_size, seq_len, units_prev_layer).
        for i in range(1, len(units_list) - 1):
            self.deep_model.append(EncoderGRU(
                units=units_list[i],
                token_length=units_list[i - 1],
                use_bias=use_bias,
                return_sequences=True,
                seed=seed,
            ))

        # Last layer can not return sequences.
        self.deep_model.append(EncoderGRU(
            units=units_list[-1],
            token_length=units_list[-2],
            use_bias=use_bias,
            return_sequences=return_sequences,
            seed=seed,
        ))

    def call(self, x):
        h = x

        for layer in self.deep_model:
            h = layer(h)

        return h
    
    def __str__(self):
        model_str = f"Multilayer RNN with {len(self.deep_model)} layers:"

        for sub_model in self.deep_model:
            model_str += f"\n\t\t\t{sub_model}"

        return model_str
    
    def get_config(self):
        return {
            "units_list": self.units_list,
            "token_length": self.token_length,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }

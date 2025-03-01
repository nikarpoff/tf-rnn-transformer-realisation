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


@tf.keras.utils.register_keras_serializable()
class CellLSTM(tf.keras.layers.Layer):
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

        # Initialize forget gate weights. Use union matrix for stacked x and h.
        self.W_f = self.add_weight(shape=(self.token_length + self.units, self.units), initializer=w_initializer, name="W_f")

        # Initialize input gate weights. Use union matrix too.
        self.W_i = self.add_weight(shape=(self.token_length + self.units, self.units), initializer=w_initializer, name="W_i")

        # Initialize cell candidate gate weights. Use union matrix.
        self.W_g = self.add_weight(shape=(self.token_length + self.units, self.units), initializer=w_initializer, name="W_g")

        # Initialize output gate weights. Use union matrix too.
        self.W_o = self.add_weight(shape=(self.token_length + self.units, self.units), initializer=w_initializer, name="W_o")

        if self.use_bias:
            self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name="b_f")
            self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name="b_i")
            self.b_g = self.add_weight(shape=(self.units,), initializer='zeros', name="b_g")
            self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name="b_o")

        super().build(input_shape)

    def call(self, x: tf.Tensor, h: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        x_h = tf.concat([x, h], axis=-1)

        i = tf.sigmoid(tf.matmul(x_h, self.W_i) + (self.b_i if self.use_bias else 0))  # what should be remembered
        f = tf.sigmoid(tf.matmul(x_h, self.W_f) + (self.b_f if self.use_bias else 0))  # what should be forgotten
        o = tf.sigmoid(tf.matmul(x_h, self.W_o) + (self.b_o if self.use_bias else 0))  # what should be added to h
        g = tf.tanh(tf.matmul(x_h, self.W_g) + (self.b_g if self.use_bias else 0))     # new context (candidate cell) 

        c_new = f * c + i * g       # update new context
        h_new = o * tf.tanh(c_new)  # count new hiddent state

        return h_new, c_new
    
    def get_config(self):
        return {
            "units": self.units,
            "token_length": self.token_length,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }


@tf.keras.utils.register_keras_serializable()
class DecoderLSTM(tf.keras.layers.Layer):
    """
    Decoder based on LSTM that used to predict translation from context and hidden state by encoder.
    """
    def __init__(self, input_length: int, output_token_length: int, max_sequence_size: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)
        self.max_sequence_size = max_sequence_size
        self.output_token_length = output_token_length
        
        # Decoder LSTM.
        self.cell = CellLSTM(units=input_length, token_length=output_token_length, use_bias=use_bias, seed=seed)

        # Dense for y prediction.
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                input_length,
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                output_token_length,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        )])

    def call(self, s_0: tf.Tensor, c_0: tf.Tensor):
        # The one step of LSTM computation.
        def loop_body(step, h, c, outputs):
            y = self.dense(h)  # prediction of word (embedding) by dense layer
            h_next, c_next = self.cell(y, h, c)  # compute new h and c based on y and previous h, c
            outputs = outputs.write(step, y)  # remember predicted word
            return step + 1, h_next, c_next, outputs

        # Outputs from LSTM (y predicted).
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # The sequence of computations. Stop when max_sequence_size was reached.
        _, _, _, outputs = tf.while_loop(
            cond=lambda step, *_: step < self.max_sequence_size,
            body=loop_body,
            loop_vars=(0, s_0, c_0, outputs)
        )

        output_sequence = outputs.stack()
        output_sequence = tf.transpose(output_sequence, [1, 0, 2])
        return output_sequence

    def get_config(self):
        return {
            "input_length": self.cell.units,
            "output_token_length": self.output_token_length,
            "max_sequence_size": self.max_sequence_size,
            "use_bias": self.cell.use_bias,
            "seed": self.cell.seed,
            "name": self.name
        }

    def __str__(self):
        return f"Decoder based on LSTM. Output shape: (None, {self.max_sequence_size}, {self.output_token_length})\n"\
               f"\tDense for y prediction: {self.dense}"
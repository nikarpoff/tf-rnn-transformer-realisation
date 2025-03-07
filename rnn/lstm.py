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
        self.y_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                input_length,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                output_token_length,
                activation='linear',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        )])

    def call(self, s_0: tf.Tensor, c_0: tf.Tensor):
        # The one step of LSTM computation.
        def loop_body(step, h, c, outputs):
            y = self.y_dense(h)  # prediction of word (embedding) by dense layer
            h_next, _ = self.cell(y, h, c)  # compute new h and c based on y and previous h, c
            outputs = outputs.write(step, y)  # remember predicted word
            return step + 1, h_next, c_0, outputs

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
               f"\tDense for y prediction: {self.y_dense}"
    

class DecoderAttentionLSTM(DecoderLSTM):
    """
    Decoder based on LSTM with attention that used to predict translation from context and hidden states by encoder.
    
    Consumes tensor h instead of vector c_0. Used h to evaluate attention weights and form context vector c.
    """
    def __init__(self, input_length: int, output_token_length: int, max_sequence_size: int, use_bias: bool, seed: int, name=None):
        super().__init__(input_length, output_token_length, max_sequence_size, use_bias, seed, name=name)
        
        # MLP for prediction e - the necessarity of every hidden state from encoder between 0 and 1.
        self.e_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                input_length,
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        )])

        # Softmax to convert alignment scores into probabilities - weights of every hidden state.
        self.a_softmax = tf.keras.layers.Softmax()

    def call(self, h: tf.Tensor, s_0: tf.Tensor):
        max_seq = self.max_sequence_size

        # The one step of decoder with attention computation.
        def loop_over_sequence(step, h, s, outputs):
            y = self.y_dense(s)  # prediction of word (embedding) by dense layer

            e_ta = tf.TensorArray(tf.float32, size=max_seq)

            # Over the h 3-d data count vector e (batch_size, sequence_size).
            _, e_ta = tf.while_loop(
                cond=lambda i, *_: i < max_seq,
                body=lambda i, ta: (i+1, ta.write(i, self.e_dense(tf.concat([h[:, i, :], s], axis=1)))),
                loop_vars=(0, e_ta)
            )

            e = e_ta.stack()  # Shape: (max_seq, batch_size, 1)
            e = tf.squeeze(e, axis=-1)  # Shape: (max_seq, batch_size)
            e = tf.transpose(e)  # Shape: (batch_size, max_seq)
            e = tf.ensure_shape(e, [None, max_seq])

            # Get vector 'a' throught softmax.
            a = self.a_softmax(e)

            # Vector c is weighted sum of hidden states with attention weights a.
            new_c = tf.reduce_sum(h * tf.expand_dims(a, axis=-1), axis=1)  # sum along sequence
            new_c = tf.ensure_shape(new_c, [None, self.cell.units])

            s_next, _ = self.cell(y, s, new_c)  # compute new s based on prev y and s, new c

            s_next = tf.ensure_shape(s_next, ([None, self.cell.units]))

            outputs = outputs.write(step, y)  # remember predicted word
            return step + 1, h, s_next, outputs

        outputs = tf.TensorArray(tf.float32,
                                 size=max_seq,
                                 element_shape=[None, self.output_token_length]
        )  # LSTM outputs

        # The sequence of computations. Stop when max_sequence_size was reached.
        _, _, _, outputs = tf.while_loop(
            cond=lambda step, *_: step < self.max_sequence_size,
            body=loop_over_sequence,
            loop_vars=(0, h, s_0, outputs),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None, None, self.cell.units]),
                tf.TensorShape([None, self.cell.units]),
                tf.TensorShape(None)
            )
        )

        output_sequence = outputs.stack()
        output_sequence = tf.transpose(output_sequence, [1, 0, 2])
        return output_sequence

    def __str__(self):
        return f"Decoder based on LSTM with attention. Output shape: (None, {self.max_sequence_size}, {self.output_token_length})\n"

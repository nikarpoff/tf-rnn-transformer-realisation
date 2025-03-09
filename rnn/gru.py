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


@tf.keras.utils.register_keras_serializable()
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
            h_next = tf.ensure_shape(h_next, [None, self.units])  # h shape is constant
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


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
class DecoderAttentionGRU(tf.keras.layers.Layer):
    """
    Decoder based on LSTM with attention that used to predict translation from context and hidden states by encoder.
    
    Consumes tensor h instead of vector c_0. Used h to evaluate attention weights and form context vector c.
    """
    def __init__(self, hidden_length: int, output_token_length: int, max_sequence_size: int, use_bias: bool, seed: int, name=None):
        super().__init__(name=name)
        self.hidden_length = hidden_length
        self.output_token_length = output_token_length
        self.max_sequence_size = max_sequence_size

        # Cause we will concatenate vectors y and c we should to initialize token length as sum of len y and len c.
        self.cell = CellGRU(units=hidden_length,
                            token_length=output_token_length + hidden_length,
                            use_bias=use_bias,
                            seed=seed
        )

        # Dense for y prediction.
        self.y_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_length,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                hidden_length,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                output_token_length,
                activation='linear',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        )])

        # MLP for prediction e - the necessarity of every hidden state from encoder between 0 and 1.
        self.e_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_length,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        ), tf.keras.layers.Dense(
                1,
                activation='linear',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                use_bias=use_bias
        )])

        # Softmax to convert alignment scores into probabilities - weights of every hidden state.
        self.a_softmax = tf.keras.layers.Softmax()

    def call(self, h: tf.Tensor, s_0: tf.Tensor):
        max_seq = self.max_sequence_size
        h_len = self.hidden_length

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

            e = tf.transpose(tf.squeeze(e_ta.stack(), axis=-1))
            e = tf.ensure_shape(e, [None, max_seq])

            # Get vector 'a' throught softmax.
            a = self.a_softmax(e)

            # Vector c is weighted sum of hidden states with attention weights a.
            attention_c = tf.reduce_sum(h * tf.expand_dims(a, axis=-1), axis=1)  # sum along sequence
            attention_c = tf.ensure_shape(attention_c, [None, h_len])

            combined_input = tf.concat([y, attention_c], axis=-1)
            s_next = self.cell(combined_input, s)  # compute new s based on prev y and s, new c

            s_next = tf.ensure_shape(s_next, ([None, h_len]))

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

    def get_config(self):
        return {
            "hidden_length": self.hidden_length,
            "output_token_length": self.output_token_length,
            "max_sequence_size": self.max_sequence_size,
            "use_bias": self.cell.use_bias,
            "seed": self.cell.seed,
            "name": self.name
        }

    def __str__(self):
        return f"Decoder based on LSTM with attention. Output shape: (None, {self.max_sequence_size}, {self.output_token_length})\n"

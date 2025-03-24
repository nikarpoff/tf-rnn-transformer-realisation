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

from rnn.gru import DeepEncoderGRU, DecoderAttentionGRU
from rnn.lstm import DecoderLSTM
from transformer.transformer import TransformerEncoder, TransformerDecoder

@tf.keras.utils.register_keras_serializable()
class Translator(tf.keras.Model):
    """
    Translation model.
    """
    def __init__(self, token_length: int, max_sequence_size: int, seed=None, name=None):
        super().__init__(name=name)
        self.token_length = token_length
        self.max_sequence_size = max_sequence_size
        self.seed = seed

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "token_length": self.token_length,
            "max_sequence_size": self.max_sequence_size,
            "seed": self.seed,
            "name": self.name
        }


class TranslatorRNN(Translator):
    """
    Translation model based on RNN
    """
    def __init__(self, encoder_units: list, token_length: int, max_sequence_size: int,
                use_bias=True, seed=None, name=None):
        super().__init__(token_length, max_sequence_size, seed, name)
        self.encoder_units = encoder_units
        self.use_bias = use_bias

        # Initialize encoder. It returns ht that can be used for decoder.
        self.encoder = DeepEncoderGRU(encoder_units,
                                      token_length=token_length,
                                      use_bias=use_bias,
                                      return_sequences=False,
                                      seed=seed)

        # Initialize dense layer to predict s0 from encoder ht output.
        self.s0_dense = tf.keras.layers.Dense(
            units=encoder_units[-1],
            use_bias=use_bias,
            name="s0_dense",
        )
        
        # Initialize decoder based on LSTM. Decoder units number is encoder's last layer units number.
        self.decoder = DecoderLSTM(input_length=encoder_units[-1],  # encoder output is decoder input.
                                   output_token_length=token_length,  # use the same token length
                                   max_sequence_size=max_sequence_size,
                                   use_bias=use_bias,
                                   seed=seed
        )

    def call(self, x):
        ht = self.encoder(x)
        
        c_0 = ht
        s_0 = self.s0_dense(ht)

        return self.decoder(s_0, c_0)

    def get_config(self):
        return {
            "encoder_units": self.encoder_units,
            "token_length": self.token_length,
            "max_sequence_size": self.max_sequence_size,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }

    def __str__(self):
        return f"Simple Translator RNN\n\tEncoder: {self.encoder}\n\tDense: {self.s0_dense}\n\tDecoder: {self.decoder}"   


class TranslatorAttentionRNN(Translator):
    """
    Translation model based on RNN with attention principle
    """
    def __init__(self, encoder_units: list, token_length: int, max_sequence_size: int,
                use_bias=True, seed=None, name=None):
        super().__init__(token_length, max_sequence_size, seed, name)
        self.encoder_units = encoder_units
        self.use_bias = use_bias

        # Initialize encoder. It returns tensor h that can be used for decoder with attention.
        self.encoder = DeepEncoderGRU(encoder_units,
                                      token_length=token_length,
                                      use_bias=use_bias,
                                      return_sequences=True,
                                      seed=seed)

        # Initialize dense layer to predict s0 from encoder h[-1] output.
        self.s0_dense = tf.keras.layers.Dense(
            units=encoder_units[-1],
            use_bias=use_bias,
            name="s0_dense",
        )
        
        # Initialize decoder based on GRU. Decoder units number is encoder's last layer units number.
        self.decoder = DecoderAttentionGRU(hidden_length=encoder_units[-1],  # encoder output is decoder input.
                                           output_token_length=token_length,  # use the same token length
                                           max_sequence_size=max_sequence_size,
                                           use_bias=use_bias,
                                           seed=seed
        )

    def call(self, x):
        h = self.encoder(x)        
        s_0 = self.s0_dense(h[:, -1, :])
        return self.decoder(h, s_0)

    def get_config(self):
        return {
            "encoder_units": self.encoder_units,
            "token_length": self.token_length,
            "max_sequence_size": self.max_sequence_size,
            "use_bias": self.use_bias,
            "seed": self.seed,
            "name": self.name
        }

    def __str__(self):
        return f"Translator RNN with attention\n\tEncoder: {self.encoder}\n\tDense: {self.s0_dense}\n\tDecoder: {self.decoder}"   


class TranslatorTransformer(Translator):
    def __init__(self, model_depth: int, mlp_units: int, token_length: int, max_sequence_size: int, heads_number: int, sos_token: tf.Tensor, eos_token: tf.Tensor, seed=None, name=None):
        super().__init__(token_length, max_sequence_size, seed, name)
        self.mlp_units = mlp_units
        self.model_depth = model_depth
        self.heads_number = heads_number

        self.encoder = TransformerEncoder(model_depth, mlp_units, token_length, max_sequence_size, heads_number, seed=seed)
        self.decoder = TransformerDecoder(model_depth, mlp_units, token_length, max_sequence_size, heads_number, seed=seed)
        
        self.final_layer = tf.keras.layers.Dense(token_length)

        self.sos_token = sos_token
        self.eos_token = eos_token

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def create_padding_mask(self, seq):
        # seq: (batch_size, seq_len)
        mask = tf.cast(tf.reduce_all(tf.math.equal(seq, 0), axis=-1), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def call(self, input, target):
        encoder_padding_mask = self.create_padding_mask(input)

        target_seq_len = tf.shape(target)[1]
        look_ahead_mask = self.create_look_ahead_mask(target_seq_len)

        z = self.encoder(input, encoder_padding_mask)
        o = self.decoder(target, z, look_ahead_mask, encoder_padding_mask)

        return self.final_layer(o)

    def predict(self, input):
        batch_size = tf.shape(input)[0]
        encoder_padding_mask = self.create_padding_mask(input)
        z = self.encoder(input, encoder_padding_mask)

        sos_token = tf.convert_to_tensor(self.sos_token, dtype=tf.float32)

        outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        outputs = outputs.write(0, tf.repeat(sos_token[tf.newaxis, :], repeats=batch_size, axis=0))

        max_iter = tf.constant(self.max_sequence_size)
        eos_token = tf.constant(self.eos_token, dtype=tf.float32)

        def cond(i, outputs, stop_flag):
            return tf.logical_and(i < max_iter, tf.logical_not(stop_flag))

        def body(i, outputs, stop_flag):
            y = tf.transpose(outputs.stack(), [1, 0, 2])
            look_ahead_mask = self.create_look_ahead_mask(i+1)

            o = self.decoder(y, z, look_ahead_mask, encoder_padding_mask)

            predictions = self.final_layer(o[:, -1:, :])

            is_eos = tf.reduce_all(tf.equal(predictions, eos_token))
            new_stop_flag = tf.logical_or(stop_flag, is_eos)

            outputs = outputs.write(i+1, predictions[:, 0, :])

            return i+1, outputs, new_stop_flag

        _, final_outputs, _ = tf.while_loop(
            cond,
            body,
            loop_vars=(
                tf.constant(0),
                outputs,
                tf.constant(False)
            )
        )

        output_sequence = tf.transpose(final_outputs.stack(), [1, 0, 2])
        return output_sequence[:, 1:, :]

    def get_config(self):
        return {
            "mlp_units": self.mlp_units,
            "model_depth": self.model_depth,
            "heads_number": self.heads_number,
            "token_length": self.token_length,
            "max_sequence_size": self.max_sequence_size,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "seed": self.seed,
            "name": self.name
        }
    
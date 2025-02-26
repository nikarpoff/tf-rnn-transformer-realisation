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

from gru import DeepEncoderGRU
from dense import Dense
from lstm import DecoderLSTM
from exception import InvalidInputException


class TranslatorRNN(tf.Module):
    """
    Translation model based on RNN
    """
    def __init__(self, encoder_units: list, token_length: int, max_sequence_size: int,
                use_bias=True, seed=None, name=None):
        super().__init__(name=name)

        # Initialize encoder. It returns ht that can be used for decoder.
        self.encoder = DeepEncoderGRU(encoder_units, token_length, use_bias=use_bias, seed=seed)

        # Initialize dense layer to predict s0 from encoder ht output.
        self.s0_dense = Dense(units=encoder_units[-1], input_length=encoder_units[-1], use_bias=use_bias, seed=seed)
        
        # Initialize decoder based on LSTM. Decoder units number is encoder's last layer units number.
        self.decoder = DecoderLSTM(input_length=encoder_units[-1], output_token_length=token_length,
                                   max_sequence_size=max_sequence_size, use_bias=use_bias, seed=seed)

    def __call__(self, x):
        ht = self.encoder(x)
        
        c_0 = ht
        s_0 = self.s0_dense(ht)

        return self.decoder(s_0, c_0)

    def __str__(self):
        # return f"Simple Translator RNN\n\tEncoder: {self.encoder}\n\tDense: {self.s0_dense}\n\tDecoder: {self.decoder}"   
        return f"Simple Translator RNN\n\tEncoder: {self.encoder}"            

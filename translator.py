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

from gru import DeepGRU
from lstm import DecoderLSTM
from exception import InvalidInputException
   

class TranslatorRNN(tf.Module):
    """
    Translation model based on RNN
    """
    def __init__(self, encoder_units: list, decoder_units: list, token_length: int, learning_rate=0.001,
                 regularization_strength=0., use_bias=True, seed=None, name=None):
        super().__init__(name=name)

        # Initialize encoder. It returns ht that can be used for decoder.
        self.encoder = DeepGRU(encoder_units, token_length, use_bias=use_bias, return_sequences=False, seed=seed)

    def __call__(self, x):
        # Input Data should be 3-dimensional.
        if x.shape.ndims != 3:
            raise InvalidInputException(x.shape.ndims)

        return self.encoder(x)

    def __str__(self):
        # return f"Simple Translator RNN\n\tEncoder: {self.encoder}\n\tDecoder: {self.decoder}"
        return f"Simple Translator RNN\n\tEncoder: {self.encoder}"

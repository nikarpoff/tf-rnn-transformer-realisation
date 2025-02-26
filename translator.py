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
import re

from gru import DeepEncoderGRU
from dense import Dense
from lstm import DecoderLSTM
from gensim.utils import simple_preprocess


@tf.keras.utils.register_keras_serializable()
class TranslatorRNN(tf.keras.Model):
    """
    Translation model based on RNN
    """
    def __init__(self, encoder_units: list, token_length: int, max_sequence_size: int,
                use_bias=True, seed=None, name=None):
        super().__init__(name=name)
        self.encoder_units = encoder_units
        self.token_length = token_length
        self.max_sequence_size = max_sequence_size
        self.use_bias = use_bias
        self.seed = seed

        # Initialize encoder. It returns ht that can be used for decoder.
        self.encoder = tf.keras.Sequential(name="encoder")
        self.encoder.add(tf.keras.Input(shape=(max_sequence_size, token_length)))

        for i in range(len(encoder_units)):
            self.encoder.add(tf.keras.layers.GRU(encoder_units[i],
                                                 use_bias=use_bias,
                                                 seed=seed,
                                                 return_sequences=True if i < len(encoder_units) - 1 else False,  # last layer returns vectors
                                                 name=f"gru_layer_{i}",
                                                 #regularization, dropout, etc,
            ))

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __str__(self):
        # return f"Simple Translator RNN\n\tEncoder: {self.encoder}\n\tDense: {self.s0_dense}\n\tDecoder: {self.decoder}"   
        return f"Simple Translator RNN\n\tEncoder: {self.encoder}"


class VectorizationTranslator():
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.clean_text_regex = re.compile(r"[^а-яА-ЯёЁa-zA-Z]")

        self.ru_to_en_embedding = vectorizer.wv["человечество"] - vectorizer.wv["humanity"]
        self.en_to_ru_embedding = vectorizer.wv["humanity"] - vectorizer.wv["человечество"]

    def _preprocess_text(self, text: str):
        text = self.clean_text_regex.sub(" ", text.lower())  # standardize
        return simple_preprocess(text, min_len=1)  # tokenize

    def translate_ru_to_en(self, text: str):
        text = self._preprocess_text(text)
        result_a = ""
        result_b = ""

        for word in text:
            translation_embedding_a = self.vectorizer.wv[word] + self.ru_to_en_embedding
            translation_embedding_b = self.vectorizer.wv[word] + self.en_to_ru_embedding

            result_a += f"{self.vectorizer.wv.similar_by_vector(translation_embedding_a, topn=1)[0][0]} "
            result_b += f"{self.vectorizer.wv.similar_by_vector(translation_embedding_b, topn=1)[0][0]} "

        print(result_a)
        print(result_b)

        return result_b

    def translate_en_to_ru(text: str):
        pass

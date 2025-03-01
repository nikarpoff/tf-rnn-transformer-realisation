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

from translator import TranslatorRNN
from rnn.lstm import DecoderLSTM, CellLSTM
from rnn.gru import DeepEncoderGRU, GRU, CellGRU
from util.preprocess import TextPreprocessing


def translate_with_rnn(string: str, preprocessing: TextPreprocessing, model_path, source_lang: str):
    try:
        loaded_model = tf.keras.models.load_model(model_path, custom_objects={"TranslatorRNN": TranslatorRNN,
                                                                              "DeepEncoderGRU": DeepEncoderGRU,
                                                                              "GRU": GRU,
                                                                              "CellGRU": CellGRU,
                                                                              "DecoderLSTM": DecoderLSTM,
                                                                              "CellLSTM": CellLSTM})
    except:
        raise Exception("There is no required model in local files! First, train rnn model!")

    example_string = tf.constant(string, dtype=tf.string)

    if (source_lang == "ru"):
        encoded_example = preprocessing.preprocess_ru_string(example_string)
        decoded_example = preprocessing.get_ru_string_from_embedding(encoded_example[0])

        encoded_translation = loaded_model.predict(encoded_example)
        decoded_translation = preprocessing.get_en_string_from_embedding(tf.constant(encoded_translation)[0])
    else:
        encoded_example = preprocessing.preprocess_en_string(example_string)
        decoded_example = preprocessing.get_en_string_from_embedding(encoded_example[0])

        encoded_translation = loaded_model.predict(encoded_example)
        decoded_translation = preprocessing.get_ru_string_from_embedding(tf.constant(encoded_translation)[0])

    print(f"Original: {decoded_example}\n")
    print(f"Translation by loaded model: {decoded_translation}")

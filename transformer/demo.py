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

from translator import TranslatorTransformer
from transformer.transformer import PositionalEncoding, MultiHeadAttentionUnit, TransformerEncoder, TransformerDecoder
from util.preprocess import TextPreprocessing

def load_transformer(model_path):
    try:
        loaded_model = tf.keras.models.load_model(model_path, custom_objects={"TranslatorTransformer": TranslatorTransformer,
                                                                              "PositionalEncoding": PositionalEncoding,
                                                                              "MultiHeadAttentionUnit": MultiHeadAttentionUnit,
                                                                              "TransformerEncoder": TransformerEncoder,
                                                                              "TransformerDecoder": TransformerDecoder})
    
        return loaded_model
    except:
        raise Exception("There is no required model in local files! First, train transformer model!")


def translate(string: str, loaded_model, preprocessing: TextPreprocessing, source_lang: str):
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


def translate_with_transformer(string: str, preprocessing: TextPreprocessing, model_path, source_lang: str):
    loaded_model = load_transformer(model_path)
    translate(string, loaded_model, preprocessing, source_lang)

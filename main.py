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

import os
import numpy as np
import tensorflow as tf

import util
from translator import TranslatorRNN
from lstm import DecoderLSTM, CellLSTM
from gru import DeepEncoderGRU, GRU, CellGRU
from preprocess import TextPreprocessing, preprocess_text


MODELS_PATH = "models"
RNN_MODELS_PATH = os.path.join(MODELS_PATH, "rnn_translators")
TOKEN_LENGTH = 300
SEQUENCE_SIZE = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20


if __name__ == '__main__':
    # Loading dataset ted_hrlr_translate/ru_to_en.
    print("\nLoading dataset...\n")
    
    train_dataset, val_dataset, test_dataset, info = util.load_dataset('ted_hrlr_translate/ru_to_en')

    print("Dataset loaded! Info:")
    print(info)

    print("Loading vectorization models...")

    try:
        ru_vectorizer, en_vectorizer = util.load_ru_en_models()
        print("\nVectorization models were loaded from local files.\n")
    except Exception:
        print("\nVectorization models not found in local files. Fitting models...\n")
        ru_vectorizer, en_vectorizer = util.get_vectorization_models(train_dataset, TOKEN_LENGTH)

    print("Vectorization models are ready! Start fitting translator RNN.\n")

    # Preporcess dataset to use FastText embeddings
    preprocessing = TextPreprocessing(ru_vectorizer, en_vectorizer, BATCH_SIZE, SEQUENCE_SIZE, TOKEN_LENGTH)

    train_data = preprocessing.preprocess_dataset(train_dataset)
    val_data = preprocessing.preprocess_dataset(val_dataset)
    test_data = preprocessing.preprocess_dataset(test_dataset)

    # Initialize model.
    translator = TranslatorRNN(encoder_units=[500, 500, 100], token_length=TOKEN_LENGTH, max_sequence_size=SEQUENCE_SIZE, seed=7)
    print(translator)

    # Fit model.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.MeanSquaredError()

    # train_model(translator, val_data, EPOCHS, optimizer, loss)
    translator.compile(optimizer=optimizer, loss=loss)
    translator.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_data)

    translator.save(os.path.join(RNN_MODELS_PATH, f"rnn-translator"))

    # Test.
    loaded_model = tf.keras.models.load_model(os.path.join(RNN_MODELS_PATH, f"rnn-translator"),
                                          custom_objects={"TranslatorRNN": TranslatorRNN,
                                                          "DeepEncoderGRU": DeepEncoderGRU,
                                                          "GRU": GRU,
                                                          "CellGRU": CellGRU,
                                                          "DecoderLSTM": DecoderLSTM,
                                                          "CellLSTM": CellLSTM})
    
    example_string = tf.constant("И вот, на рассвете ты не заметил, как начался новый день", dtype=tf.string)
    encoded_example = preprocessing.preprocess_ru_string(example_string)

    print(preprocessing.get_ru_string_from_embedding(encoded_example[0]))

    translation = translator(encoded_example)
    translation_from_loaded = loaded_model(encoded_example)

    print(f"Translation: {preprocessing.get_en_string_from_embedding(tf.constant(translation)[0])}")
    print(f"Translation by loaded model: {preprocessing.get_en_string_from_embedding(tf.constant(translation_from_loaded)[0])}")

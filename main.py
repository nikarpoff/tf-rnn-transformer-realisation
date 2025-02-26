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
import tensorflow_datasets as tfds

from translator import TranslatorRNN
from preprocess import TextPreprocessing, get_vectorization_model, preprocess_text
from gensim.models import FastText


MODELS_PATH = "models"
VECTORIZATION_MODELS_PATH = os.path.join(MODELS_PATH, "vectorization")
RNN_MODELS_PATH = os.path.join(MODELS_PATH, "rnn_translators")
VECTORIZATION_RU_MODEL_PATH = os.path.join(VECTORIZATION_MODELS_PATH, "ru_model.bin")
VECTORIZATION_EN_MODEL_PATH = os.path.join(VECTORIZATION_MODELS_PATH, "en_model.bin")
TOKEN_LENGTH = 300
SEQUENCE_SIZE = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20


def train_step(model, optimizer, loss_func, x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        loss = loss_func(y_batch, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

def train_model(model, dataset, epochs, optimizer, loss_func):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for batch, (x_batch, y_batch) in enumerate(dataset):
            loss = train_step(model, optimizer, loss_func, x_batch, y_batch)
            epoch_loss += loss

            if batch % 100 == 0:
                print(f"\tBatch {batch}, Loss: {loss.numpy():.4f}")

        print(f"Epoch loss: {epoch_loss.numpy() / len(dataset)}")


if __name__ == '__main__':
    # Loading dataset ted_hrlr_translate/ru_to_en.
    print("\nLoading dataset...\n")
    dataset, info = tfds.load('ted_hrlr_translate/ru_to_en', with_info=True, as_supervised=True)
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']

    print("Dataset loaded! Info:")
    print(info)

    # Initializing Word2Vec text vectorization models.
    if not os.path.isdir(MODELS_PATH):
        os.mkdir(MODELS_PATH)
        os.mkdir(VECTORIZATION_MODELS_PATH)
        os.mkdir(RNN_MODELS_PATH)

    if os.path.isfile(VECTORIZATION_RU_MODEL_PATH) and os.path.isfile(VECTORIZATION_EN_MODEL_PATH):
        print("\nVectorization models were found in local files. Loading models...\n")
        ru_vectorizer = FastText.load(VECTORIZATION_RU_MODEL_PATH)
        en_vectorizer = FastText.load(VECTORIZATION_EN_MODEL_PATH)
    else:
        print("\nVectorization models not found in local files. Fitting models...\n")
        ru_vectorizer, en_vectorizer = get_vectorization_model(train_dataset, TOKEN_LENGTH, VECTORIZATION_RU_MODEL_PATH, VECTORIZATION_EN_MODEL_PATH)

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

    train_model(translator, val_data, EPOCHS, optimizer, loss)

    # Test.
    example_string = tf.constant("И вот, на рассвете ты не заметил, как начался новый день", dtype=tf.string)
    encoded_example = preprocessing.preprocess_ru_string(example_string)

    print(preprocessing.get_ru_string_from_embedding(encoded_example[0]))

    translation = translator(encoded_example)

    print(preprocessing.get_en_string_from_embedding(tf.constant(translation)[0]))

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
import tensorflow as tf
import tensorflow_datasets as tfds

from translator import TranslatorRNN
from preprocess import fit_vectorization_model
from gensim.models import FastText


VECTORIZATION_MODELS_PATH = "vectorization-models"
VECTORIZATION_RU_MODEL_PATH = f"{VECTORIZATION_MODELS_PATH}/word2vec_ru.model"
VECTORIZATION_EN_MODEL_PATH = f"{VECTORIZATION_MODELS_PATH}/word2vec_en.model"
TOKEN_LENGTH = 100


if __name__ == '__main__':
    # Loading dataset ted_hrlr_translate/ru_to_en.
    print("\nLoading dataset...\n")
    dataset, info = tfds.load('ted_hrlr_translate/ru_to_en', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    print("Dataset loaded! Info:")
    print(info)

    # Initializing Word2Vec text vectorization models.
    if not os.path.isdir(VECTORIZATION_MODELS_PATH):
        os.mkdir(VECTORIZATION_MODELS_PATH)

    if os.path.isfile(VECTORIZATION_RU_MODEL_PATH) and os.path.isfile(VECTORIZATION_EN_MODEL_PATH):
        print("\nVectorization models were found in local files. Loading models...\n")
        ru_vectorizer = FastText.load(VECTORIZATION_RU_MODEL_PATH)
        en_vectorizer = FastText.load(VECTORIZATION_EN_MODEL_PATH)
    else:
        print("\nVectorization models not found in local files. Fitting models...\n")
        ru_vectorizer, en_vectorizer = fit_vectorization_model(train_dataset, TOKEN_LENGTH, VECTORIZATION_RU_MODEL_PATH, VECTORIZATION_EN_MODEL_PATH)

    print("Vectorization models are ready! Start fitting translator RNN.\n")

    print(ru_vectorizer.wv["ахахахахах"])

    # Initialize model.
    translator = TranslatorRNN(encoder_units=[100, 50, 100], token_length=TOKEN_LENGTH, max_sequence_size=15, seed=7)
    print(translator)

    # Fit model.
    # translator.fit()

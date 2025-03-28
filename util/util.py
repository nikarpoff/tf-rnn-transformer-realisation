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


import tensorflow_datasets as tfds
from concurrent.futures import ThreadPoolExecutor
from gensim.models import FastText
from util.preprocess import preprocess_text


def load_dataset(dataset_name: str):
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset['train'], dataset['validation'], dataset['test'], info


def load_ru_en_models(ru_path, en_path):
    ru_vectorizer = FastText.load(ru_path)
    en_vectorizer = FastText.load(en_path)

    return ru_vectorizer, en_vectorizer

def load_ru_en_model(model_path):
    return FastText.load(model_path)

def get_vectorization_models(dataset, token_length, ru_model_save_path, en_model_save_path):
    """
    Returns vectorization FastText models trained on dataset.

    Saves trained model.
    :returns: packed ru_model and en_model
    """
    def process_sentence(ru, en):
        ru = preprocess_text(ru)
        ru.insert(0, "sos")
        ru.append("eos")

        en = preprocess_text(en)
        en.insert(0, "sos")
        en.append("eos")
        return ru, en

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sentence, *zip(*dataset)))

    ru_sentences, en_sentences = zip(*results)

    ru_model = FastText(sentences=ru_sentences, vector_size=token_length, window=5, min_count=1, workers=12)
    en_model = FastText(sentences=en_sentences, vector_size=token_length, window=5, min_count=1, workers=12)

    ru_model.save(ru_model_save_path)
    en_model.save(en_model_save_path)

    return ru_model, en_model

def get_ru_en_vectorization_model(dataset, token_length, ru_en_model_save_path):
    """
    Returns vectorization FastText model trained on dataset (on both of ru and en examples).

    Saves trained model.
    :returns: packed ru_model and en_model
    """

    def process_sentence(ru, en):
        return preprocess_text(ru), preprocess_text(en)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sentence, *zip(*dataset)))

    sentences = [sentence for pair in results for sentence in pair]
    
    ru_en_model = FastText(sentences=sentences, vector_size=token_length, window=5, min_count=1, workers=12)

    ru_en_model.save(ru_en_model_save_path)

    return ru_en_model

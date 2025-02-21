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

import re
import tensorflow as tf
from gensim.utils import simple_preprocess
from gensim.models import FastText


def preprocess_text(text):
    """
    Remove all symbols except letters of ru and eng alphabet.

    Tokenize text.
    """
    text = text = tf.compat.as_text(text.numpy())
    text = re.sub(r"[^а-яА-Яa-zA-Z]", " ", text.lower())  # standardize
    return simple_preprocess(text, min_len=1)  # tokenize

def fit_vectorization_model(dataset, token_length, ru_model_save_path, en_model_save_path):
    ru_sentences = []
    en_sentences = []
    
    for ru, en in dataset:
        ru_sentences.append(preprocess_text(ru))
        en_sentences.append(preprocess_text(en))
    
    ru_model = FastText(sentences=ru_sentences, vector_size=token_length, window=5, min_count=1, workers=12)
    en_model = FastText(sentences=en_sentences, vector_size=token_length, window=5, min_count=1, workers=12)

    ru_model.save(ru_model_save_path)
    en_model.save(en_model_save_path)

    return ru_model, en_model

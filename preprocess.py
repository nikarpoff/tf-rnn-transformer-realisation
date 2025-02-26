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
import numpy as np
import tensorflow as tf
from gensim.utils import simple_preprocess


CLEAN_TEXT_REGEX = re.compile(r"[^а-яА-ЯёЁa-zA-Z]")

def preprocess_text(text):
    """
    Remove all symbols except letters of ru and eng alphabet.

    Tokenize text.
    """
    text = tf.get_static_value(text).decode("utf-8")
    text = CLEAN_TEXT_REGEX.sub(" ", text.lower())  # standardize
    return simple_preprocess(text, min_len=1)  # tokenize

def sentence_to_embedding(sentence, model, sentence_length, token_length):
    """
    Creates embedding with constant length from sentence.
    
    Returns matrix with shape (sentence_length, token_length).
    """
    tokens = preprocess_text(sentence)

    if not tokens:
        return np.zeros(shape=(sentence_length, token_length), dtype=np.float32)

    # Use embeddings from model and transform result to required shape
    word_vectors = np.zeros((sentence_length, token_length), dtype=np.float32)
    word_vectors[:len(tokens)] = np.vstack([model.wv[word] for word in tokens[:sentence_length]])

    return word_vectors


class TextPreprocessing:
    def __init__(self, ru_model, en_model, batch_size, sentence_length, token_length):
        self.ru_model = ru_model
        self.en_model = en_model
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.token_length = token_length

        self.STOP_TOKEN = tf.zeros(shape=token_length, dtype=np.float32)

    def preprocess_dataset(self, dataset):
        """
        Preprocess strings list like dataset to tensors with shape (batch_size, sentence_length, token_length).

        Uses embeddings models for text vectorization.
        """
        def wrap_func(ru, en):
            return (sentence_to_embedding(ru, self.ru_model, self.sentence_length, self.token_length),
                    sentence_to_embedding(en, self.en_model, self.sentence_length, self.token_length))
    
        def tf_wrap(ru, en):
            return tf.numpy_function(wrap_func, [ru, en], [tf.float32, tf.float32])
    
        return (dataset.map(tf_wrap, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

    def preprocess_ru_string(self, string):
        return tf.expand_dims(sentence_to_embedding(string, self.ru_model, self.sentence_length, self.token_length), axis=0)

    def preprocess_en_string(self, string):
        return tf.expand_dims(sentence_to_embedding(string, self.en_model, self.sentence_length, self.token_length), axis=0)
    
    def get_ru_string_from_embedding(self, embedding):
        string_result = ""

        for i in range(self.sentence_length):
            if tf.reduce_all(tf.abs(embedding[i] - self.STOP_TOKEN) < 1e-5):
                return string_result
            
            string_result += self.ru_model.wv.similar_by_vector(embedding[i].numpy(), topn=1)[0][0]
            string_result += " "

        return string_result

    def get_en_string_from_embedding(self, embedding):
        string_result = ""

        for i in range(self.sentence_length):
            if tf.reduce_all(tf.abs(embedding[i] - self.STOP_TOKEN) < 1e-5):
                return string_result
            
            string_result += self.en_model.wv.similar_by_vector(embedding[i].numpy(), topn=1)[0][0]
            string_result += " "

        return string_result

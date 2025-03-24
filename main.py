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
import time
import argparse
import tensorflow as tf

import util.util
import rnn.demo
import transformer.demo
from util.preprocess import TextPreprocessing
from translator import TranslatorRNN, TranslatorAttentionRNN, TranslatorTransformer


MODELS_PATH = "models"
VECTORIZATION_MODELS_PATH = os.path.join(MODELS_PATH, "vectorization")
VECTORIZATION_RU_MODEL_PATH = os.path.join(VECTORIZATION_MODELS_PATH, "ru_model.bin")
VECTORIZATION_EN_MODEL_PATH = os.path.join(VECTORIZATION_MODELS_PATH, "en_model.bin")
VECTORIZATION_RU_EN_MODEL_PATH = os.path.join(VECTORIZATION_MODELS_PATH, "ru_en_model.bin")

RNN_MODELS_PATH = os.path.join(MODELS_PATH, "rnn_translators")
RNN_RU_MODEL_PATH = os.path.join(RNN_MODELS_PATH, "rnn-ru-to-en-translator")
RNN_EN_MODEL_PATH = os.path.join(RNN_MODELS_PATH, "rnn-en-to-ru-translator")

ATTENTION_RU_MODEL_PATH = os.path.join(RNN_MODELS_PATH, "attention-ru-to-en-translator")
ATTENTION_EN_MODEL_PATH = os.path.join(RNN_MODELS_PATH, "attention-en-to-ru-translator")

TRANSFORMER_MODELS_PATH = os.path.join(MODELS_PATH, "transformer_translators")
TRANSFORMER_RU_MODEL_PATH = os.path.join(TRANSFORMER_MODELS_PATH, "transformer-ru-to-en-translator")
TRANSFORMER_EN_MODEL_PATH = os.path.join(TRANSFORMER_MODELS_PATH, "transformer-en-to-ru-translator")

TOKEN_LENGTH = 100
SEQUENCE_SIZE = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 5

SEED = 7


def cli_arguments_preprocess():
    parser = argparse.ArgumentParser(description="Machine translation powered by ANN")
    
    parser.add_argument("--task", required=True, 
                      choices=["train", "test", "translate"],
                      help="The operating mode: train/test/translate")
    
    parser.add_argument("--model", required=True,
                      choices=["rnn", "attention", "transformer", "vectorization"],
                      help="Model: rnn/transformer. For train only embedding models you can use 'vectorization' with task==train")

    parser.add_argument("--lang", 
                      choices=["ru", "en"],
                      help="Language of source text (required for translate)")
    
    parser.add_argument("text", nargs="?", help="Text to be translated")

    args = parser.parse_args()

    if args.model == "vectorization" and args.task != "train":
        parser.error("You can only use this model for 'train' tasks.")

    if not args.lang and args.model != "vectorization":
        parser.error("For any task except train embedding models you must provide --lang")

    if args.task == "translate":
        if not args.text:
            parser.error("For translation, you need to provide the text")

    return args.task, args.model, args.lang, args.text

def create_file_structure():
    if not os.path.isdir(MODELS_PATH):
        os.mkdir(MODELS_PATH)
        
    if not os.path.isdir(VECTORIZATION_MODELS_PATH):
        os.mkdir(VECTORIZATION_MODELS_PATH)

    if not os.path.isdir(RNN_MODELS_PATH):
        os.mkdir(RNN_MODELS_PATH)

    if not os.path.isdir(TRANSFORMER_MODELS_PATH):
        os.mkdir(TRANSFORMER_MODELS_PATH)

def load_dataset(silence=True):
    # Loading dataset ted_hrlr_translate/ru_to_en.
    if silence:
        train_dataset, val_dataset, test_dataset, _ = util.util.load_dataset("ted_hrlr_translate/ru_to_en")
        return train_dataset, val_dataset, test_dataset
    else:
        print("\nLoading dataset...\n")

        train_dataset, val_dataset, test_dataset, info = util.util.load_dataset("ted_hrlr_translate/ru_to_en")
    
        print("Dataset loaded! Info:")
        print(info)

        return train_dataset, val_dataset, test_dataset

def load_vectorization_models():
    print("Loading vectorization models...")

    try:
        ru_vectorizer, en_vectorizer = util.util.load_ru_en_models(VECTORIZATION_RU_MODEL_PATH, VECTORIZATION_EN_MODEL_PATH)
        print("\nVectorization models were loaded from local files.\n")
    except Exception:
        print("\nVectorization models not found in local files.\n")
        return None, None
        
    return ru_vectorizer, en_vectorizer

def fit_vectorization_models(dataset, token_length):
    print("Fitting ru and en vectorization models...")
    ru_vectorizer, en_vectorizer = util.util.get_vectorization_models(dataset, token_length, VECTORIZATION_RU_MODEL_PATH, VECTORIZATION_EN_MODEL_PATH)

    print("Vectorization models are ready!\n")
    return ru_vectorizer, en_vectorizer

def fit_translator_rnn(translator, train_data, val_data, optimizer, loss, save_path):
    translator.build(input_shape=(BATCH_SIZE, SEQUENCE_SIZE, TOKEN_LENGTH))
    translator.summary()

    # Fit model.
    translator.compile(optimizer=optimizer, loss=loss)
    translator.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_data)

    translator.save(save_path)

def fit_transformer(translator, train_data, val_data, optimizer, loss_object, save_path):
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    translator.compile(optimizer=optimizer, loss=loss_object)

    @tf.function
    def train_step(input, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            predictions = translator(input, target_input)
            loss = loss_function(target_real, predictions)

        gradients = tape.gradient(loss, translator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, translator.trainable_variables))

        train_loss(loss)

    @tf.function
    def validate_step(input, target):
        predictions = translator.predict(input)
        loss = loss_function(target, predictions)

        val_loss(loss)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        val_loss.reset_states()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        for (batch, (input, target)) in enumerate(train_data):
            train_step(input, target)

            if batch % 50 == 0:
                print(f'\tBatch {batch} Loss {train_loss.result():.4f}')

        for input, target in val_data:
            validate_step(input, target)

        print(f'Train Loss {train_loss.result():.4f}; Validation loss {val_loss.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    
    translator.save(save_path)


def main():
    task, model, lang, text = cli_arguments_preprocess()

    # Dataset required for train and test tasks.
    if task == "train" or task == "test":
        train_dataset, val_dataset, test_dataset = load_dataset(silence=False)

    # Load embedding models.
    ru_vectorizer, en_vectorizer = load_vectorization_models()

    # Embedding models were not loaded.
    if not ru_vectorizer or not en_vectorizer:
        # In task 'translate' we can not use dataset -> we should throw exception
        if task == "translate":
            raise Exception("You want to use vectorization models, but they were not found in the local files. Try to train models by command \n\tpython main.py --task=train --model=vectorization.")
        
        create_file_structure()  # here we know that vectorization model not found in local files -> there no any models and dirs for them
        ru_vectorizer, en_vectorizer = fit_vectorization_models(train_dataset, TOKEN_LENGTH)


    # Vectorization models were learned!
    if model == "vectorization" and task == "train":
        return

    # Preporcess dataset with FastText embeddings model.
    preprocessing = TextPreprocessing(ru_vectorizer, en_vectorizer, BATCH_SIZE, SEQUENCE_SIZE, TOKEN_LENGTH)

    # Remember paths to models.
    if lang == "ru":
        rnn_path = RNN_RU_MODEL_PATH
        attention_rnn_path = ATTENTION_RU_MODEL_PATH
        transformer_path = TRANSFORMER_RU_MODEL_PATH
    if lang == "en":
        rnn_path = RNN_EN_MODEL_PATH
        attention_rnn_path = ATTENTION_EN_MODEL_PATH
        transformer_path = TRANSFORMER_EN_MODEL_PATH

    # Task translate -> dataset not required, model can be loaded
    if task == "translate":
        if model == "rnn":
            rnn.demo.translate_with_rnn(text, preprocessing, rnn_path, lang)
        elif model == "attention":
            rnn.demo.translate_with_attention_rnn(text, preprocessing, attention_rnn_path, lang)
        elif model == "transformer":
            transformer.demo.translate_with_transformer(text, preprocessing, transformer_path, lang)
        else:
            raise Exception(f"Unknown model: {model}")

        return
    
    # Preprocess dataset.
    if lang == "ru":
        train_data = preprocessing.preprocess_ru_to_en_dataset(train_dataset)
        val_data = preprocessing.preprocess_ru_to_en_dataset(val_dataset)
        test_data = preprocessing.preprocess_ru_to_en_dataset(test_dataset)
        target_vectorizer = en_vectorizer
    if lang == "en":
        train_data = preprocessing.preprocess_en_to_ru_dataset(train_dataset)
        val_data = preprocessing.preprocess_en_to_ru_dataset(val_dataset)
        test_data = preprocessing.preprocess_en_to_ru_dataset(test_dataset)
        target_vectorizer = ru_vectorizer

    # Specify optimizer and loss function.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.CosineSimilarity()    
        
    if task == "test":
        if model == "rnn":
            print("TODO...")
        if model == "transformer":
            print("TODO...")

        return

    # There we have "train" task
    if task == "train":
        encoder_units = [300, 200]

        if model == "rnn":
            translator = TranslatorRNN(encoder_units=encoder_units, token_length=TOKEN_LENGTH, max_sequence_size=SEQUENCE_SIZE, seed=SEED)
            fit_translator_rnn(translator, train_data, val_data, optimizer, loss, rnn_path)
        
        elif model == "attention":
            translator = TranslatorAttentionRNN(encoder_units=encoder_units, token_length=TOKEN_LENGTH, max_sequence_size=SEQUENCE_SIZE, seed=SEED)
            fit_translator_rnn(translator, train_data, val_data, optimizer, loss, attention_rnn_path)

        elif model=="transformer":
            model_depth = 300
            mlp_units = 512
            heads_number = 96

            sos_token = target_vectorizer.wv["sos"]
            eos_token = target_vectorizer.wv["eos"]

            translator = TranslatorTransformer(model_depth, mlp_units, TOKEN_LENGTH, SEQUENCE_SIZE, heads_number, sos_token, eos_token, seed=SEED)
            fit_transformer(translator, train_data, val_data, optimizer, loss, transformer_path)
        else:
            raise Exception(f"Unknown model: {model}")

        
    else:
        raise Exception(f"Unknown task: {task}")

if __name__ == "__main__":
    main()

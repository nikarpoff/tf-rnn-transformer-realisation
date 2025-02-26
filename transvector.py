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

import util
from translator import VectorizationTranslator

MODELS_PATH = "models"
TOKEN_LENGTH = 300
SEQUENCE_SIZE = 15


if __name__ == '__main__':
    # Loading dataset ted_hrlr_translate/ru_to_en.
    print("\nLoading dataset...\n")
    
    train_dataset, _, _, info = util.load_dataset('ted_hrlr_translate/ru_to_en')

    print("Dataset loaded! Info:")
    print(info)

    print("Loading vectorization model...")
    
    try:
        ru_en_vectorizer = util.load_ru_en_model()
        print("\nVectorization model loaded from local files.\n")
    except Exception:
        print("\nVectorization model not found in local files. Fitting model...\n")
        ru_en_vectorizer = util.get_ru_en_vectorization_model(train_dataset, TOKEN_LENGTH)

    print("Vectorization model are ready!\n")

    # Initialize model.
    translator = VectorizationTranslator(ru_en_vectorizer)

    # Test.
    print(translator.translate_ru_to_en("И вот, на рассвете ты не заметил, как начался новый день"))

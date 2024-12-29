import tensorflow as tf
from rnn import TranslatorRNN

if __name__ == '__main__':
    translator = TranslatorRNN(5, 5, 3)

    print(translator(tf.constant([[[3, 3, 5]]], dtype=tf.float32)))

    print(translator)

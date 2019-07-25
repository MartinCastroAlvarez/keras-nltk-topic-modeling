import sys
import logging

import numpy as np

import keras.utils
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


logger = logging.getLogger(__name__)

# Ref:
# https://towardsdatascience.com/text-classification-in-keras-part-1-a-simple-reuters-news-classifier-9558d34d01d3

if __name__ == "__main__":

    # Printing logs to console.
    # Reference: https://stackoverflow.com/questions/14058453
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Running Main handler.
    logger.setLevel(logging.DEBUG)

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # REf:
    # https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa

    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    # restore np.load for future normal usage
    np.load = np_load_old

    print('# of Training Samples: {}'.format(len(x_train)))
    print('# of Test Samples: {}'.format(len(x_test)))

    num_classes = max(y_train) + 1
    print('# of Classes: {}'.format(num_classes))

    index_to_word = {}
    for key, value in word_index.items():
        index_to_word[value] = key

    print(' '.join([index_to_word[x] for x in x_train[0]]))
    print(y_train[0])

    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(x_train[0])
    print(len(x_train[0]))

    print(y_train[0])
    print(len(y_train[0]))

    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.metrics_names)

    batch_size = 32
    epochs = 3

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

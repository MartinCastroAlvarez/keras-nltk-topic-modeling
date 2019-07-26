"""
Script to create a prediction model.
"""

import os
import sys
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt

import keras.utils
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional, LSTM

# Initializing constants.
LOG_LEVEL = logging.DEBUG
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
MAX_WORDS = 10000
BATCH_SIZE = 300
EPOCHS = 5
LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER_FUNCTION = "adam"
ACCURACY_METRIC = "accuracy"
BINARY = "binary"
RELU = "relu"
SOFTMAX = "softmax"
MODEL_PATH = os.path.join("models", "{}_model.json")
WEIGHTS_PATH = os.path.join("models", "{}_weights.h5")
PLOT_PATH = os.path.join("models", "{}_loss.png")
TOKENIZER_PATH = os.path.join("models", "{}_tokenizer.pkl")
LOSS = "loss"
VAL_LOSS = "val_loss"

# Initializing logger.
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# Avoiding this issue:
# https://stackoverflow.com/questions/55890813
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Loading dataset
logger.debug("Loading dataset. | sf_split=%s", TEST_SPLIT)
train_set, test_set = reuters.load_data(num_words=None, test_split=TEST_SPLIT)
x_train, y_train = train_set
x_test, y_test = test_set
logger.debug("Dataset Loaded. | sf_train=%s | sf_test=%s", len(x_train), len(x_test))

# Loading the words index.
word_index = reuters.get_word_index()
logger.debug("Word index loaded. | sf_index=%s", len(word_index))
# Indexing all labels in the dataset.
word_by_id_index = {}
for key, value in word_index.items():
    word_by_id_index[value] = key
logger.debug("Indexed words by ID. | sf_index=%s", len(word_by_id_index))

# Avoiding this issue:
# https://stackoverflow.com/questions/55890813
# Restoring np.load for future normal usage
np.load = np_load_old

# Fetching distinct classes.
total_labels = max(y_train) + 1
logger.debug("Labels detected. | sf_train=%s | sf_test=%s | sf_labels=%s",
             len(y_train), len(y_test), total_labels)

# Tokenizing dataset using a One-Hot Encoder.
tokenizer = Tokenizer(num_words=MAX_WORDS)
x_train = tokenizer.sequences_to_matrix(x_train, mode=BINARY)
x_test = tokenizer.sequences_to_matrix(x_test, mode=BINARY)
logger.debug("Dataset tokenized. | sf_train=%s | sf_test=%s", len(x_train), len(x_test))
y_train = keras.utils.to_categorical(y_train, total_labels)
y_test = keras.utils.to_categorical(y_test, total_labels)
logger.debug("Labels tokenized. | sf_train=%s | sf_test=%s", len(y_train), len(y_test))

# Defining the Neural Network.
logger.debug("Creating Neural Network.")
model = Sequential()
# Ref:
# https://keras.io/layers/embeddings/
# https://arxiv.org/pdf/1301.3781.pdf
# https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
# model.add(Embedding(MAX_WORDS, 32))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dense(1, activation='sigmoid'))
# model.add(Activation('relu'))
# model.add(Embedding(total_labels, 32))
model.add(Dense(512, input_shape=(MAX_WORDS,)))
model.add(Activation(RELU))
model.add(Dropout(0.5))
model.add(Dense(total_labels))
model.add(Activation(SOFTMAX))

# Compiling Neural Network.
# Setting loss function and back-propagation optimizer.
model.compile(loss=LOSS_FUNCTION,
              metrics=[ACCURACY_METRIC],
              optimizer=OPTIMIZER_FUNCTION)

# Training the model.
# Using mini batches.
history = model.fit(x_train, y_train, verbose=1, batch_size=BATCH_SIZE,
                    epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

# Evaluating the performance of the model.
score = model.evaluate(x_test, y_test, verbose=1, batch_size=BATCH_SIZE)
logger.debug("Performance evaluated. | sf_loss=%s | sf_accuracy=%s", score[0], score[1])

# Serializing model to JSON.
# Serializing weights to HDF5.
model_json = model.to_json()
model_path = MODEL_PATH.format(int(10000 * score[1]))
weights_path = WEIGHTS_PATH.format(int(10000 * score[1]))
with open(model_path, "w") as json_file:
    json_file.write(model_json)
model.save_weights(weights_path)
logger.debug("Model saved. | sf_model=%s | sf_weights=%s", model_path, weights_path)

# Persisting the Tokenizer.
tokenizer_path = TOKENIZER_PATH.format(int(10000 * score[1]))
with open(tokenizer_path, 'wb') as file_buffer:
    pickle.dump(tokenizer, file_buffer, protocol=pickle.HIGHEST_PROTOCOL)
logger.debug("Tokenizer persisted. | sf_path=%s", tokenizer_path)

# Plotting results.
plot_path = PLOT_PATH.format(int(10000 * score[1]))
loss = history.history[LOSS]
val_loss = history.history[VAL_LOSS]
epochs = range(1, len(loss) + 1)
plt.clf()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(plot_path)
logger.debug("Epochs plotted. | sf_path=%s", plot_path)

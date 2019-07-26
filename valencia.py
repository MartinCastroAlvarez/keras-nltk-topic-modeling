"""
Script to make real-time predictions.
"""

import os
import sys
import logging

import requests
import pickle
import html2text

from slugify import slugify

from keras.datasets import reuters
from keras.models import model_from_json

# Initializing constants.
LOG_LEVEL = logging.DEBUG
MODEL_PATH = os.path.join("models", "{}_model.json")
WEIGHTS_PATH = os.path.join("models", "{}_weights.h5")
TOKENIZER_PATH = os.path.join("models", "{}_tokenizer.pkl")
BINARY = "binary"
TOP_CLASSES = 40
HTML_PATH = os.path.join("html")
PREDICTIONS_PATH = os.path.join("predictions")
URLS_PATH = os.path.join("urls.csv")

# Loading labels.
LABELS = [
    'cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
    'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
    'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
    'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
    'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead',
]

# Initializing logger.
logger = logging.getLogger(__name__)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# Define functio nto slugify strings.
def normalize(my_string: str):
    return "".join(
        x for x in slugify(my_string)
        if x.isalnum
        and x not in ("-", ".")
    )

# Loading script version.
version = sys.argv[1] if len(sys.argv) > 1 else ""
if not version:
    raise ValueError("Model version is required.")
logger.debug("Version initilized. | sf_value=%s", version)

# Detecting model path.
model_path = MODEL_PATH.format(version)
if not os.path.isfile(model_path):
    raise OSError("File not found:", model_path)
logger.debug("Model path found. | sf_path=%s", model_path)

# Detecting weights path.
weights_path = WEIGHTS_PATH.format(version)
if not os.path.isfile(weights_path):
    raise OSError("Weights not found:", weights_path)
logger.debug("Weights path found. | sf_path=%s", weights_path)

# Loading prediction model from file.
# load json and create model
with open(model_path, 'r') as file_buffer:
    loaded_model_json = file_buffer.read()
model = model_from_json(loaded_model_json)
model.load_weights(weights_path)
logger.debug("Prediction model loaded. | sf_model=%s", model)

# Loading the Tokenizer.
tokenizer_path = TOKENIZER_PATH.format(version)
if not os.path.isfile(tokenizer_path):
    raise OSError("File not found:", tokenizer_path)
with open(tokenizer_path, 'rb') as file_buffer:
    tokenizer = pickle.load(file_buffer)
logger.debug("Tokenizer loaded. | sf_tokenizer=%s", tokenizer)

# Loading the words index.
word_index = reuters.get_word_index()
logger.debug("Word index loaded. | sf_index=%s", len(word_index))
# Indexing all labels in the dataset.
id_by_word_index = {}
for key, value in word_index.items():
    id_by_word_index[key] = value
logger.debug("Indexed words by ID. | sf_index=%s", len(id_by_word_index))

# Initializing HTML processor.
html_processor = html2text.HTML2Text()

# Reading URL or all.
if len(sys.argv) > 2:
    urls = sys.argv[2:]
else:
    with open(URLS_PATH, "r") as file_buffer:
        urls = (
            line.split(",")[0]
            for line in file_buffer.read().split("\n")
        )
        urls = [
            url
            for url in urls
            if url.startswith("http")
        ]

# Parsing each URL individually.
for url in urls:
    logger.debug("Parsing URL. | sf_url=%s", url)

    # Checking if URL is loca 
    cache_path = os.path.join(HTML_PATH, normalize(url))
    if os.path.isfile(cache_path):

        # Loading file from the local file system.
        logger.debug("Loading local content. | sf_url=%s", url)
        with open(cache_path, "r") as file_buffer:
            text = file_buffer.read()
        logger.debug("Local content loaded. | sf_url=%s", url)

    else:

        # Loading URL content.
        logger.debug("Downloading content. | sf_url=%s", url)
        response = requests.get(url)
        logger.debug("Respose obtained. | sf_response=%s", response)
        if response.status_code != 200:
            logger.error("Error detected. | sf_response=%s", response)
            raise RuntimeError("Bad request:", url)
        logger.debug("Downloaded content. | sf_url=%s", url)
        text = response.text

        # SAving to cache.
        with open(cache_path, "w") as file_buffer:
            file_buffer.write(text)
        logger.debug("Cache saved. | sf_url=%s", cache_path)

    # Converting HTML to text.
    text = html_processor.handle(text).lower()
    logger.debug("Extracted raw text. | sf_text=%s", len(text))

    # Tokenizing text.
    x_new = [[
        id_by_word_index[w]
        for w in text.split(" ")
        if w in id_by_word_index
    ]]
    logger.debug("Text indexed. | sf_text=%s", len(x_new))
    x_new = tokenizer.sequences_to_matrix(x_new, mode=BINARY)
    logger.debug("Text tokenized. | sf_text=%s", len(x_new))

    # Make predictions.
    results = model.predict(x_new)
    logger.debug("Text classified. | sf_predictions=%s", len(results[0]))

    # Extract classes from results.
    score_by_class = {
        LABELS[i]: r
        for i, r in enumerate(results[0])
    }
    logger.debug("Classes extracted. | sf_classes=%s", len(score_by_class))

    # Sorting classes by score.
    sorted_classes = sorted(score_by_class, key=lambda x: score_by_class[x], reverse=True)
    logger.debug("Classes sorted. | sf_classes=%s", len(sorted_classes))

    # Saving results to the predictions path.
    predictions_path = os.path.join(PREDICTIONS_PATH, normalize(url))
    logger.debug("Savings results. | sf_path=%s", predictions_path)
    with open(predictions_path, "w") as file_buffer:

        # Getting top classes. Printing report.
        print("[{}]".format(url), file=file_buffer)
        for name in sorted_classes[:TOP_CLASSES]:
            score = score_by_class[name]
            logger.debug("Top class. | sf_class=%s | sf_score=%s", name, score)
            print("- {}: {}".format(name, score), file=file_buffer)

    # End of URLs
    logger.debug("Text classified. | sf_results=%s", predictions_path)

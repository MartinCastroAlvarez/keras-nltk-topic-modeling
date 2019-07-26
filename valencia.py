"""
Script to make real-time predictions.
"""

import os
import sys
import logging

from keras.models import model_from_json

# Initializing constants.
LOG_LEVEL = logging.DEBUG
MODEL_PATH = os.path.join("model-{}.json")
WEIGHTS_PATH = os.path.join("model-{}.h5")

# Initializing logger.
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# Loading script arguments.
version = sys.argv[1] if len(sys.argv) > 1 else ""
url = sys.argv[2] if len(sys.argv) > 2 else ""
if not version:
    raise ValueError("Model version is required.")
if not url:
    raise ValueError("Prediction URL is required.")
logger.debug("Version initilized. | sf_value=%s", version)
logger.debug("URL initilized. | sf_value=%s", url)

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

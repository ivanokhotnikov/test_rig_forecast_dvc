import os

from google.cloud import storage

LOCAL_DATA_PATH = os.path.join('..', 'data')
IMAGES_PATH = os.path.join('.', 'images')
MODELS_PATH = os.path.join('.', 'models')
PREDICTIONS_PATH = os.path.join('.', 'predictions')

STORAGE_CLIENT = storage.Client()

FOLDS = 5
SEED = 42
VERBOSITY = 1
OPTIMIZATION_TIME_BUDGET = 5 * 60 * 60
TIME_STEPS = 120
EARLY_STOPPING = 10
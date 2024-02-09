import os

SAVED_WEIGHTS_DIR = 'saved_weights'
SAVED_MODEL_DIR = 'saved_models'
DATA_DIR = 'datasets'

for dir in [SAVED_MODEL_DIR, DATA_DIR, SAVED_WEIGHTS_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def load_data(path):
    with open(path) as f:
        return f.read()
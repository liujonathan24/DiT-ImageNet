import os

# Data loading from https://www.kaggle.com/code/liucong12601/dinov2-imagenet-training

BASE_PATH = "./data/"

TRAIN_PATHS = [
    os.path.join(BASE_PATH, 'train.X1'),
    os.path.join(BASE_PATH, 'train.X2'),
    os.path.join(BASE_PATH, 'train.X3'),
    os.path.join(BASE_PATH, 'train.X4'),
]
VALID_PATH = os.path.join(BASE_PATH, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')
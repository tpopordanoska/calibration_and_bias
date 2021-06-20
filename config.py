import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
SAVED_PREDICTIONS_DIR = './data/predictions'

full_images_path = 'data/MICCAI_BraTS_2018_Data_Training'
SEP_IMAGE_PATH = os.path.join(RESULTS_DIR, "separate_images")

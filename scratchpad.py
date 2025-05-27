import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tensorflow_datasets as tfds
# import tensorflow as tf
import os
import librosa
from scipy import signal
import pickle

Current_dataset = 'dtu_td_cmaa_50ov'
file_path = os.path.join("logs", 'history', f'train_val_params_{Current_dataset}_cont.pkl')
if os.path.exists(file_path):
    # Read the existing file if it exists
    with open(file_path, "rb") as pickle_file:
        old_Train_val_params = pickle.load(pickle_file)

        print(sorted(old_Train_val_params.keys()))
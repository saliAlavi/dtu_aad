import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
# import tensorflow_models as tfm
import tensorflow_hub as hub
import os
from scipy import signal
from models import *
import json
import altair as alt
import io
import plotly.graph_objects as go
import pickle

"""
To solve the cudalib problem:https://github.com/pytorch/pytorch/issues/85773
"""

BATCH_SIZE = 32
EPOCHS = 40

latent_dim = 64
best_val_acc = {}
all_val_acc = {}
train_val_acc={}
curr_fold=0
curr_subject = 'exc'
curr_rep= 0
g_curr_fold = 0


class SaveParamsIterCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self,batch, logs=None):
        train_acc_name = f'train_acc_sub-{curr_subject}_fold-{curr_fold}_rep-{curr_rep}'
        if train_acc_name not in train_val_acc.keys():
            train_val_acc[train_acc_name]=[]
        train_val_acc[train_acc_name].append(logs['acc'])

        train_loss_name = f'train_loss_sub-{curr_subject}_fold-{curr_fold}_rep-{curr_rep}'
        if train_loss_name not in train_val_acc.keys():
            train_val_acc[train_loss_name] = []
        train_val_acc[train_loss_name].append(logs['loss'])

        train_cont_loss_name = f'train_cont_loss_sub-{curr_subject}_fold-{curr_fold}_rep-{curr_rep}'
        if train_cont_loss_name not in train_val_acc.keys():
            train_val_acc[train_cont_loss_name] = []
        train_val_acc[train_cont_loss_name].append(logs['loss_ctr'])


    def on_epoch_end(self, epoch, logs=None):
        if logs['val_acc']>best_val_acc[curr_fold]:
            best_val_acc[curr_fold]=logs['val_acc']

        val_acc_name = f'val_cont_acc_sub-{curr_subject}_fold-{curr_fold}_rep-{curr_rep}'
        if val_acc_name not in train_val_acc.keys():
            train_val_acc[val_acc_name] = []
        train_val_acc[val_acc_name].append(logs['val_acc'])

        val_loss_name = f'val_loss_sub-{curr_subject}_fold-{curr_fold}_rep-{curr_rep}'
        if val_loss_name not in train_val_acc.keys():
            train_val_acc[val_loss_name] = []
        train_val_acc[val_loss_name].append(logs['val_loss'])


    def on_train_end(selfself, epoch, logs=None):
        if g_curr_fold not in list(all_val_acc):
            all_val_acc[g_curr_fold] = []

        all_val_acc[g_curr_fold].append(best_val_acc[curr_fold])
        print(f"The overall accuracy is {np.average(np.fromiter(best_val_acc.values(), dtype=float))}")

        with open(os.path.join("logs", 'history', 'train_val_params_[per_sub].pkl'), "wb") as pickle_file:
            pickle.dump(train_val_acc, pickle_file)


class SaveBestAccCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Your custom code here
        if logs['val_acc']>best_val_acc[curr_fold]:
            best_val_acc[curr_fold]=logs['val_acc']

    def on_train_end(selfself, epoch, logs=None):
        if g_curr_fold not in list(all_val_acc):
            all_val_acc[g_curr_fold] = []

        all_val_acc[g_curr_fold].append(best_val_acc[curr_fold])
        print(f"The overall accuracy is {np.average(np.fromiter(best_val_acc.values(), dtype=float))}")


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, save_dir, save_best_only=False):
        super(SaveModelCallback, self).__init__()
        self.model = model
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        save_dir = self.save_dir + f'acc{val_acc:.2f}'
        if self.save_best_only:
            if val_acc is None:
                raise ValueError("If save_best_only=True, you must provide the validation loss.")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(save_dir)
                print(f"Saved best model to {save_dir}")
        else:

            self.model.save_weights(save_dir)
            print(f"Saved model to {save_dir}")


def feature_extraction(sample):
    x = sample['eeg']
    x = x.numpy()
    eeg_welch=np.zeros((65, 66))

    for ch in range(66):
        eeg_welch[:, ch]=signal.welch(x[:,ch])

    eeg_welch= tf.convert_to_tensor(eeg_welch, dtype=tf.float32)
    return {'eeg':eeg_welch, 'attr_lr':x['attr_lr']}

def map_stft(x):
    frame_lenth= 10
    frame_step=1
    fft_length=32
    data = x['data']
    curr_data = data[:, 0]
    curr_data = tf.signal.stft(curr_data, frame_length=frame_lenth, frame_step=frame_step, fft_length=fft_length)
    shape = (16,76, 17)
    data_real = tf.zeros([*shape, 64])
    data_imag = tf.zeros([*shape, 64])
    for ch in range(64):
        curr_data = data[:, ch]
        curr_data = tf.signal.stft(curr_data, frame_length=frame_lenth, frame_step=frame_step, fft_length=fft_length)
        data_real[..., ch] = tf.math.real(curr_data)
        data_imag[..., ch] = tf.math.imag(curr_data)
    data_stft = {'data_f_real': data_real, 'data_f_imag':data_imag}
    return {**data_stft, **x}

if __name__ == '__main__':
    train_splits = ['train[20%:]', 'train[:20%]+train[40%:]', 'train[:40%]+train[60%:]', 'train[:60%]+train[80%:]',
                    'train[:80%]']
    test_splits = ['train[:20%]', 'train[20%:40%]', 'train[40%:60%]', 'train[60%:80%]', 'train[80%:]']

    reps=2
    subjects = [2]

    ds_train = None
    ds_test = None

    for subject in subjects:
        dataset_name = f'dtu_td_cmaa_50ov/5s_sub{subject}'
        for curr_fold,(train_split, test_split) in enumerate(zip(train_splits, test_splits)):
            for rep in range(reps):
                curr_rep = rep
                g_curr_fold = curr_fold
                print(f'Fold:{curr_fold} Rep:{rep} Subject:{subject}')
                best_val_acc[curr_fold]=0
                del(ds_train)
                del(ds_test)
                (ds_train, ds_test), dataset_info = tfds.load(dataset_name, split=[train_split, test_split], shuffle_files=False,with_info=True,)

                # ds_train= ds_train.cache().shuffle(dataset_info.splits[train_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
                # ds_test = ds_test.cache().shuffle(dataset_info.splits[test_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

                ds_train = ds_train.cache().shuffle(dataset_info.splits[train_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
                ds_test = ds_test.cache().shuffle(dataset_info.splits[test_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

                x_eeg_csp = ds_train.map(lambda x: x['eeg_csp'])
                x_audio_m_ds = ds_train.map(lambda x: x['audio_m_ds'])
                x_audio_f_ds = ds_train.map(lambda x: x['audio_f_ds'])

                model = AttnCrossAudioEEGSimsiam()

                model.normalizer_eeg.adapt(x_eeg_csp)
                model.normalizer_audio_m.adapt(x_audio_m_ds)
                model.normalizer_audio_f.adapt(x_audio_f_ds)

                opt_ctr = tf.keras.optimizers.Adam(learning_rate=0.001)
                opt_cls = tf.keras.optimizers.Adam(learning_rate=0.001)
                opt = keras.optimizers.Adamax(learning_rate=0.001)
                model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

                #Callbacks
                num_training_samples = len(ds_train)
                steps = EPOCHS * (num_training_samples // BATCH_SIZE)
                lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03, decay_steps=steps)

                log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,min_lr=0.001)
                save_best_acc_callback = SaveBestAccCallback()
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                save_dir = os.path.join('saved_models')
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, f'model_{subject}')
                save_model_callback = SaveModelCallback(model, save_path, save_best_only=True)

                save_params_callback=SaveParamsIterCallback()
                #End Callbacks

                history = model.fit(
                            ds_train,
                            epochs=EPOCHS,
                            shuffle=True,
                            validation_data=(ds_test),
                            callbacks=[reduce_lr_callback, save_best_acc_callback, tensorboard_callback, save_model_callback,save_params_callback])

                print(all_val_acc)

                # with open(os.path.join("logs", 'history', 'val_acc_sbj.pkl'), "wb") as pickle_file:
                #     pickle.dump(all_val_acc, pickle_file)

        print(history)




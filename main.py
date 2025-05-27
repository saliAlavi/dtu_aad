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
import tensorflow_models as tfm
import tensorflow_hub as hub
import os
from scipy import signal
from models import *
import json
import pandas as pd
import altair as alt
import io
import plotly.graph_objects as go
import pickle

"""
To solve the cudalib problem:https://github.com/pytorch/pytorch/issues/85773
"""

BATCH_SIZE = 32
EPOCHS = 100

latent_dim = 64
best_val_acc = {}
all_val_acc = {}
curr_fold=0
curr_subject = 1

class SaveBestAccCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Your custom code here


        if logs['val_acc']>best_val_acc[curr_fold]:
            best_val_acc[curr_fold]=logs['val_acc']

    def on_train_end(selfself, epoch, logs=None):
        if curr_subject not in list(all_val_acc.keys()):
            all_val_acc[curr_subject] = []

        all_val_acc[curr_subject].append(best_val_acc[curr_fold])
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
        # save_dir = self.save_dir + f'acc{val_acc:.2f}.hp5'
        save_dir = self.save_dir + f'acc{val_acc:.2f}'
        if self.save_best_only:
            if val_acc is None:
                raise ValueError("If save_best_only=True, you must provide the validation loss.")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(save_dir)
                # self.model.save(save_dir)
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

    # dataset_name='kuleuven_stft/5s_sub16'
    # dataset_name = 'kuleuven_td/5s_sub12'
    # dataset_name = 'ku_luven_td/5s_sub16'
    # dataset_name = 'kuleuven_stft/5s_all'
    # dataset_name = 'ku_luven_tfds/5s_sub1'

    subjects = list(range(1, 19))
    train_splits = ['train[20%:]', 'train[:20%]+train[40%:]', 'train[:40%]+train[60%:]', 'train[:60%]+train[80%:]','train[:80%]']
    test_splits = ['train[:20%]', 'train[20%:40%]','train[40%:60%]','train[60%:80%]','train[80%:]']


    # subjects = range(12, 19)
    subjects = [1]
    single_split=True
    split_select = 3
    if single_split:
        train_splits = [train_splits[split_select]]
        test_splits = [test_splits[split_select]]
    # train_splits = ['train[20%:]']
    # test_splits = ['train[:20%]']
    # train_splits = ['train[:20%]+train[40%:]']
    # test_splits = ['train[20%:40%]']
    # epochs = 1

    for subject in subjects:
        curr_subject= subject
        # dataset_name = f'dtu_td_cmaa/5s_sub{subject}'
        # dataset_name = f'dtu_td_cmaa_50ov_2s/2s_sub{subject}'
        dataset_name = f'dtu_td_cmaa_50ov/5s_sub{subject}'
        for curr_fold,(train_split, test_split) in enumerate(zip(train_splits, test_splits)):
            print(f'Subject:{subject} Fold:{curr_fold}')
            best_val_acc[curr_fold]=0
            (ds_train, ds_test), dataset_info = tfds.load(dataset_name, split=[train_split, test_split], shuffle_files=False,with_info=True,)

            ds_train= ds_train.cache().shuffle(dataset_info.splits[train_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.cache().shuffle(dataset_info.splits[test_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

            x_eeg_csp = ds_train.map(lambda x: x['eeg_csp'])
            x_audio_m_ds = ds_train.map(lambda x: x['audio_m_ds'])
            x_audio_f_ds = ds_train.map(lambda x: x['audio_f_ds'])

            # model =AttnCrossAudioEEG()
            # model = AttnCrossAudioEEGContr()
            model = AttnCrossAudioEEGSimsiam()



            model.normalizer_eeg.adapt(x_eeg_csp)
            model.normalizer_audio_m.adapt(x_audio_m_ds)
            model.normalizer_audio_f.adapt(x_audio_f_ds)

            customCallback =SaveBestAccCallback()
            opt_ctr = tf.keras.optimizers.Adam(learning_rate=0.001)
            opt_cls = tf.keras.optimizers.Adam(learning_rate=0.001)
            opt = keras.optimizers.Adamax(learning_rate=0.001)
            model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

            model.fit(ds_train,
                      epochs=1,
                      shuffle=True,
                      validation_data=(ds_test),
                      callbacks=[])
            checkpoint_path = os.path.join('saved_models', 'model_sub18_spl4_acc0.96.hp5')
            model.load_weights(checkpoint_path)


            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                                         patience=5, min_lr=0.001)
            num_training_samples = len(ds_train)
            steps = EPOCHS * (num_training_samples // BATCH_SIZE)
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.03, decay_steps=steps
            )
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            save_dir = os.path.join('saved_models')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, f'model_sub{subject}_spl{split_select}_')
            save_model_callback = SaveModelCallback(model, save_path, save_best_only=True)
            history = model.fit(ds_train,
                        epochs=EPOCHS,
                        shuffle=True,
                        validation_data=(ds_test),
                        callbacks=[reduce_lr, customCallback,tensorboard_callback, save_model_callback])

        with open(os.path.join("logs", 'history', 'val_acc_sbj.pkl'), "wb") as pickle_file:
            pickle.dump(all_val_acc, pickle_file)

    print(history)


    print(all_val_acc)
    print(f'mean: {np.mean(all_val_acc[curr_subject])}')
    df = pd.DataFrame({
    'Date': [],
    'Open': [],
    'High': [],
    'Low': [],
    'Close': []})
    df.index.name = 'Date'
    for subject in subjects:
        arr = all_val_acc[subject]
        min = np.min(arr)
        max = np.max(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        new_row = {'Date':f'{subject}', 'Open':mean+std/2, 'Close':mean-std/2, 'High':max, 'Low':min}
        df.loc[len(df)] = new_row

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    save_path = os.path.join('logs', 'figs', 'val_acc_per_subjects.png')
    fig.write_image(save_path)

    df = pd.DataFrame({
        'Date': [],
        'Open': [],
        'High': [],
        'Low': [],
        'Close': []})
    df.index.name = 'Date'
    arr = np.concatenate(list(all_val_acc.values()), axis=0)
    min = np.min(arr)
    max = np.max(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    new_row = {'Date': f'{subject}', 'Open': mean + std / 2, 'Close': mean - std / 2, 'High': max, 'Low': min}
    df.loc[len(df)] = new_row
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    save_path = os.path.join('logs', 'figs', 'val_acc_overall.png')
    fig.write_image(save_path)









    #
    # bytes_io = io.BytesIO()
    # bytes_io.seek(0)
    # file_writer = tf.summary.create_file_writer(log_dir)
    #
    # some_obj_worth_noting = {
    #     "tfds_training_data": {
    #         "name": "mnist",
    #         "split": "train",
    #         "shuffle_files": "True",
    #     },
    #     "keras_optimizer": {
    #         "name": "Adagrad",
    #         "learning_rate": "0.001",
    #         "epsilon": 1e-07,
    #     },
    #     "hardware": "Cloud TPU",
    # }
    #
    #
    # # TODO: Update this example when TensorBoard is released with
    # # https://github.com/tensorflow/tensorboard/pull/4585
    # # which supports fenced codeblocks in Markdown.
    # def pretty_json(hp):
    #     json_hp = json.dumps(hp, indent=2)
    #     return "".join("\t" + line for line in json_hp.splitlines(True))
    #
    #
    # markdown_text = """
    # ### Markdown Text
    #
    # TensorBoard supports basic markdown syntax, including:
    #
    #     preformatted code
    #
    # **bold text**
    #
    # | and | tables |
    # | ---- | ---------- |
    # | among | others |
    # """
    #
    #
    # def gen_plot():
    #     import io
    #     """Create a pyplot plot and save to buffer."""
    #     plt.figure()
    #     plt.plot([1, 2])
    #     plt.title("test")
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     buf.seek(0)
    #     return buf
    #
    # with file_writer.as_default():
    #     image = tf.image.decode_png(gen_plot().getvalue(), channels=4)
    #     image = tf.expand_dims(image, 0)
    #     summary_op = tf.summary.image("plot test", image, step=0)
    #     tf.summary.text("run_params", pretty_json(some_obj_worth_noting), step=0)
    #     tf.summary.text("markdown_jubiliee", markdown_text, step=0)
    #     tf.summary.histogram(name='weights_histogram', data=np.arange(10), step=0)
    #

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
EPOCHS = 8

latent_dim = 64
best_val_acc = {}
All_val_acc = {}
Train_val_params={}
Current_dataset = 'dtu_td_cmaa_50ov_05s'
Current_subject = 'sub1'
Curr_fold=0
Curr_rep= 0
# Curr_subject = 1



class SaveParamsIterCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self,batch, logs=None):
        train_acc_name = f'train_acc_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if train_acc_name not in Train_val_params.keys():
            Train_val_params[train_acc_name]=[]
        Train_val_params[train_acc_name].append(logs['acc'])

        train_loss_name = f'train_loss_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if train_loss_name not in Train_val_params.keys():
            Train_val_params[train_loss_name] = []
        Train_val_params[train_loss_name].append(logs['loss'])

        train_cont_loss_name = f'train_cont_loss_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if train_cont_loss_name not in Train_val_params.keys():
            Train_val_params[train_cont_loss_name] = []
        Train_val_params[train_cont_loss_name].append(logs['loss_ctr'])


    def on_epoch_end(self, epoch, logs=None):
        if logs['val_acc']>best_val_acc[Current_subject]:
            best_val_acc[Current_subject]=logs['val_acc']

        train_acc_name = f'train_acc_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if train_acc_name not in Train_val_params.keys():
            Train_val_params[train_acc_name] = []
        Train_val_params[train_acc_name].append(logs['acc'])

        val_acc_name = f'val_acc_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if val_acc_name not in Train_val_params.keys():
            Train_val_params[val_acc_name] = []
        Train_val_params[val_acc_name].append(logs['val_acc'])

        val_loss_name = f'val_loss_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if val_loss_name not in Train_val_params.keys():
            Train_val_params[val_loss_name] = []
        Train_val_params[val_loss_name].append(logs['val_loss'])

        val_cont_loss_name = f'val_cont_loss_{Current_dataset}_{Current_subject}_fold-{Curr_fold}_rep-{Curr_rep}'
        if val_cont_loss_name not in Train_val_params.keys():
            Train_val_params[val_cont_loss_name] = []
        Train_val_params[val_cont_loss_name].append(logs['loss_ctr'])

        file_path = os.path.join("logs", 'history', f'train_val_params_{Current_dataset}_cont.pkl')
        if os.path.exists(file_path):
            # Read the existing file if it exists
            with open(file_path, "rb") as pickle_file:
                try:
                    old_Train_val_params = pickle.load(pickle_file)
                    # print(old_Train_val_params.keys())
                except EOFError:
                    print("Couldn't read the old file.")
                    old_Train_val_params = {}  # If the file is empty, initialize as an empty dictionary
        else:
            old_Train_val_params = {}  # If the file doesn't exist, initialize as an empty dictionary

        Train_val_params.update(old_Train_val_params)
        with open(file_path, "wb") as pickle_file:
            pickle.dump(Train_val_params, pickle_file)


    def on_train_end(self, epoch, logs=None):
        if Current_subject not in All_val_acc.keys():
            All_val_acc[Current_subject] = []


        All_val_acc[Current_subject].append(best_val_acc[Current_subject])
        print(f"The overall accuracy is {np.average(np.fromiter(best_val_acc.values(), dtype=float))}")

        

    # def on_train_begin(self, logs=None  ):
    #     fn = os.path.join("logs", 'history', f'train_val_params_{Current_dataset}.pkl')
    #     if os.path.exists(fn):
    #         with open(fn, 'rb') as f:
    #             All_val_acc = pickle.load(f)



class SaveBestAccCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Your custom code here
        if logs['val_acc']>best_val_acc[Curr_fold]:
            best_val_acc[Curr_fold]=logs['val_acc']

    def on_train_end(selfself, epoch, logs=None):
        if Curr_fold not in list(All_val_acc):
            All_val_acc[Curr_fold] = []

        All_val_acc[Curr_fold].append(best_val_acc[Curr_fold])
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
        save_dir = self.save_dir + f'{Current_dataset}_{Current_subject}_fold-{Curr_fold}'
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


if __name__ == '__main__':
    dataset_names = ['dtu_td_cmaa_50ov', 'dtu_td_cmaa_50ov_2s', 'dtu_td_cmaa_50ov_05s']
    dataset_sub_names = [f'sub{i}' for i in range(1,17)]+['all']+[f'all_e{i}' for i in range(1,19)]
	
    reps=1

    dataset_names = ['dtu_td_cmaa_50ov_05s']
    file_path = os.path.join("logs", 'history', f'train_val_params_{Current_dataset}_cont.pkl')
    with open(file_path, "rb+") as pickle_file:
            try:
                old_Train_val_params = pickle.load(pickle_file)
                prefix = 'all_e'
                idx=1
                while(f'train_acc_{Current_dataset}_{prefix+str(idx)}_fold-0_rep-{reps-1}' in old_Train_val_params.keys()): idx+=1
                Train_val_params.update(old_Train_val_params)
                if idx>1:
                    idx-=1
                    print(f'Starting from subject {idx}')
                # f'train_acc_{Current_dataset}_{Current_subject}_fold-0_rep-0'
                # processed_subjects = map(int, old_Train_val_params.keys())
            except EOFError:
                print("Couldn't read the old file. Training from scratch.")
                old_Train_val_params = {}  # If the file is empty, initialize as an empty dictionary
    dataset_sub_names = [f'all_e{i}' for i in range(idx,19)]

    
    single_split=False
    subjects = [6]
    split_select = 3

    if single_split:
        train_splits = [train_splits[split_select]]
        test_splits = [test_splits[split_select]]

    # with open(os.path.join("logs", 'history', f'train_val_params_{Current_dataset}.pkl'), "rb") as pickle_file:
    #         pickle.dump(Train_val_params, pickle_file)


    # dataset_name = f'dtu_td_cmaa_50ov/5s_all'
    # dataset_name = f'dtu_td_cmaa_50ov_2s/all_t'
    ds_train = None
    ds_test = None
    for dataset_name in dataset_names:
        for dataset_sub_name in dataset_sub_names:
            dataset_name_full = f'{dataset_name}/{dataset_sub_name}'

            Current_dataset = dataset_name
            Current_subject = dataset_sub_name
            if 'e' in dataset_sub_name:
                train_splits =['train']
                test_splits = ['test']
            else:
                train_splits = ['train[20%:]', 'train[:20%]+train[40%:]', 'train[:40%]+train[60%:]', 'train[:60%]+train[80%:]', 'train[:80%]']
                test_splits = ['train[:20%]', 'train[20%:40%]', 'train[40%:60%]', 'train[60%:80%]', 'train[80%:]']

            best_val_acc = {}
            for Curr_fold,(train_split, test_split) in enumerate(zip(train_splits, test_splits)):
                    for rep in range(reps):
                        Curr_rep = rep
                        Curr_fold = Curr_fold
                        print(f'DS:{dataset_name_full} Fold:{Curr_fold} Rep:{rep}')
                        best_val_acc[Current_subject]=0

                        del (ds_train)
                        del (ds_test)
                        (ds_train, ds_test), dataset_info = tfds.load(dataset_name_full, split=[train_split, test_split], shuffle_files=False,with_info=True,)

                        ds_train= ds_train.shuffle(dataset_info.splits[train_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
                        ds_test = ds_test.shuffle(dataset_info.splits[test_split].num_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

                        x_eeg_csp = ds_train.map(lambda x: x['eeg_csp'])
                        x_audio_m_ds = ds_train.map(lambda x: x['audio_m_ds'])
                        x_audio_f_ds = ds_train.map(lambda x: x['audio_f_ds'])

                        model = AttnCrossAudioEEGContr()

                        model.normalizer_eeg.adapt(x_eeg_csp)
                        model.normalizer_audio_m.adapt(x_audio_m_ds)
                        model.normalizer_audio_f.adapt(x_audio_f_ds)

                        customCallback =SaveBestAccCallback()
                        opt_ctr = tf.keras.optimizers.Adam(learning_rate=0.001)
                        opt_cls = tf.keras.optimizers.Adam(learning_rate=0.001)
                        opt = keras.optimizers.Adamax(learning_rate=0.001)
                        model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

                        # Callbacks
                        num_training_samples = len(ds_train)
                        steps = EPOCHS * (num_training_samples // BATCH_SIZE)
                        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03,
                                                                                  decay_steps=steps)

                        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                                                                  patience=5, min_lr=0.001)
                        save_best_acc_callback = SaveBestAccCallback()
                        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                        save_dir = os.path.join('saved_models')
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'model_')
                        save_model_callback = SaveModelCallback(model, save_path, save_best_only=True)

                        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=os.path.join(save_dir, 'best_model_ku_5s.h5'),       
                                            monitor='val_acc',             
                                            save_best_only=True,            
                                            mode='max',                     
                                            verbose=1                       
                                        )

                        save_params_callback = SaveParamsIterCallback()
                        # End Callbacks

                        history = model.fit(
                                    ds_train,
                                    epochs=EPOCHS,
                                    shuffle=True,
                                    validation_data=(ds_test),
                                    callbacks=[reduce_lr_callback, tensorboard_callback, save_params_callback])
                        model.save_weights(os.path.join(save_dir, 'best_model_ku_5s.h5'))
                        print(All_val_acc)

                        # with open(os.path.join("logs", 'history', 'val_acc_sbj.pkl'), "wb") as pickle_file:
                        #     pickle.dump(All_val_acc, pickle_file)

    print(history)



    # all_val_acc_max = {}
    # for i, k in enumerate(list(all_val_acc[curr_subject].keys())):
    #     curr_val_acc = all_val_acc[curr_subject][k]
    #     if i ==0:
    #         all_val_acc_max[curr_subject]=[]
    #     all_val_acc_max[curr_subject].append(np.max(curr_val_acc))
    #
    # print(f'All accuracies:{all_val_acc}')
    # print(f'mean: {np.mean(all_val_acc_max[curr_subject])}')
    # # print(f'mean: {np.mean(all_val_acc[curr_subject])}')
    # df = pd.DataFrame({
    # 'Date': [],
    # 'Open': [],
    # 'High': [],
    # 'Low': [],
    # 'Close': []})
    # df.index.name = 'Date'
    # for subject in subjects:
    #     arr = all_val_acc_max[subject]
    #     min = np.min(arr)
    #     max = np.max(arr)
    #     mean = np.mean(arr)
    #     std = np.std(arr)
    #     new_row = {'Date':f'{subject}', 'Open':mean+std/2, 'Close':mean-std/2, 'High':max, 'Low':min}
    #     df.loc[len(df)] = new_row
    #
    # fig = go.Figure(data=[go.Candlestick(x=df['Date'],
    #                                      open=df['Open'],
    #                                      high=df['High'],
    #                                      low=df['Low'],
    #                                      close=df['Close'])])
    # save_path = os.path.join('logs', 'figs', 'val_acc_all.png')
    # fig.write_image(save_path)
    #
    # df = pd.DataFrame({
    #     'Date': [],
    #     'Open': [],
    #     'High': [],
    #     'Low': [],
    #     'Close': []})
    # df.index.name = 'Date'
    # arr = np.concatenate(list(all_val_acc_max.values()), axis=0)
    # min = np.min(arr)
    # max = np.max(arr)
    # mean = np.mean(arr)
    # std = np.std(arr)
    # new_row = {'Date': f'{subject}', 'Open': mean + std / 2, 'Close': mean - std / 2, 'High': max, 'Low': min}
    # df.loc[len(df)] = new_row
    # fig = go.Figure(data=[go.Candlestick(x=df['Date'],
    #                                      open=df['Open'],
    #                                      high=df['High'],
    #                                      low=df['Low'],
    #                                      close=df['Close'])])
    # save_path = os.path.join('logs', 'figs', 'val_acc_overall.png')
    # fig.write_image(save_path)

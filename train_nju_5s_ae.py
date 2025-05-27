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
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
"""
To solve the cudalib problem:https://github.com/pytorch/pytorch/issues/85773
"""

BATCH_SIZE = 100
EPOCHS = 200

latent_dim = 64
best_val_acc = {}
All_val_acc = {}
Train_val_params={}
Current_dataset = 'dtu_td_cmaa_50ov'
Current_subject = 'sub1'
Curr_fold=0
Curr_rep= 0
# Curr_subject = 1



class SaveBestValAcc(tf.keras.callbacks.Callback):
    def __init__(self, save_best=False, filepath="best_val_acc.pkl"):
        super().__init__()
        self.filepath = filepath
        self.best_val_acc = 0.0
        self.save_best = save_best
        
        # Load previous best if exists
        if os.path.exists(self.filepath):
            with open(self.filepath, "rb") as f:
                self.best_val_acc = pickle.load(f)
                # print(f"Loaded best validation accuracy: {self.best_val_acc:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        val_acc = logs.get("val_acc")  # 'val_accuracy' is the correct key in Keras
        if self.save_best:
            if val_acc is not None and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                with open(self.filepath, "wb") as f:
                    pickle.dump(self.best_val_acc, f)
                # print(f"Epoch {epoch+1}: New best val_acc: {self.best_val_acc:.4f} saved!")
            else:
                pass
                # print(f"Epoch {epoch+1}: No improvement in val_acc ({val_acc})")
        else:
            with open(self.filepath, "wb") as f:
                    pickle.dump(val_acc, f)


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, save_dir, save_best_only=True):
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

    

class SaveUMap(tf.keras.callbacks.Callback):
    def __init__(self, model, save_dir, dataset, sub_idx, idx, save_best_only=True):
        super(SaveUMap, self).__init__()
        self.model = model
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.dataset = dataset
        self.sub_idx = sub_idx
        self.idx = idx
        self.best_val_acc=0

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        data_sample = next(iter(self.dataset))
        latent_vectors=model.latent(data_sample)
        scaler = StandardScaler()
        latent_vectors_scaled = scaler.fit_transform(latent_vectors)

        # Apply UMAP for dimensionality reduction (2D visualization)
        umap_2d = umap.UMAP(n_components=2, random_state=42)
        latent_2d = umap_2d.fit_transform(latent_vectors_scaled)
        y_labels=data_sample['att_lr']
        # Plot the embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_labels, s=5, alpha=0.7, cmap='bwr')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Projection of Latent Representations')
        if val_acc>self.best_val_acc:
            self.best_val_acc=val_acc
            plt.savefig(os.path.join(self.save_dir, f'umap_sub{self.sub_idx}_idx{self.idx}'), dpi=300, bbox_inches='tight')

def merge_datasets_by_sub(dataset_name, sub_inds):
    ds_list = []
    total_size=0
    for sub_ind in sub_inds:
        dataset_name_full = f'{dataset_name}/s{sub_ind:02d}'
        cur_ds, dataset_info = tfds.load(dataset_name_full, split='train', shuffle_files=False,with_info=True,)
        # print(cur_ds)
        cur_ds = cur_ds.shuffle(dataset_info.splits['train'].num_examples)
        total_size+=dataset_info.splits['train'].num_examples
        ds_list.append(cur_ds)

    combined = tf.data.Dataset.from_tensor_slices(ds_list)
    combined = combined.interleave(
        lambda x: x,
        cycle_length=2,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    combined = combined.shuffle(buffer_size=total_size)

    return combined

if __name__ == '__main__':
    
    dataset_sub_names = [f'sub{i}' for i in range(1,17)]+['all']+[f'all_e{i}' for i in range(1,19)]
	
    reps=1

    dataset_names = ['nju_5s']
    file_path = os.path.join("logs", 'history', f'train_val_params_{Current_dataset}_cont.pkl')
    idx=1
    with open(file_path, 'a') as file:
        pass
    with open(file_path, "rb") as pickle_file:
            try:
                old_Train_val_params = pickle.load(pickle_file)
                prefix = 'all_e'
                
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

    dataset_sub_names = [f'sub{i}' for i in range(idx,19)]

    dataset_sub_names=['s03']
    single_split=False
    subjects = [6]
    split_select = 3


    ds_train = None
    ds_test = None
    all_sub_list =[2,3,4,6,7,8,9,12,13,14,15,16,17,18,19,21,22,23,25,26,27]
    ds_train_subs = [all_sub_list[1:]]
    ds_test_subs = [[2] for _ in range(len(ds_train_subs))]

    for i, (ds_train_sub, ds_test_sub) in enumerate(zip(ds_train_subs, ds_test_subs)):
        best_val_acc = {}
        print(f'train {ds_train_sub}, test:{ds_test_sub}')
        dataset_name=f'nju_5s'
        ds_train = merge_datasets_by_sub(dataset_name, ds_train_sub)
        ds_test  = merge_datasets_by_sub(dataset_name, ds_test_sub)

        ds_train= ds_train.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

        x_eeg_csp = ds_train.map(lambda x: x['eeg_csp'])
        x_audio_m_ds = ds_train.map(lambda x: x['audio_l_ds'])
        x_audio_f_ds = ds_train.map(lambda x: x['audio_r_ds'])

        model = AttnCrossAudioEEGSimsiamNJU()

        model.normalizer_eeg.adapt(x_eeg_csp)
        model.normalizer_audio_m.adapt(x_audio_m_ds)
        model.normalizer_audio_f.adapt(x_audio_f_ds)
        

        #plot
        # Standardize (optional but recommended for UMAP)
        # data_sample = next(iter(ds_test))
        # latent_vectors=model.latent(data_sample)
        # scaler = StandardScaler()
        # latent_vectors_scaled = scaler.fit_transform(latent_vectors)

        # # Apply UMAP for dimensionality reduction (2D visualization)
        # umap_2d = umap.UMAP(n_components=2, random_state=42)
        # latent_2d = umap_2d.fit_transform(latent_vectors_scaled)
        # y_labels=data_sample['lr']
        # print(f'label {y_labels}')
        # # Plot the embeddings
        # plt.figure(figsize=(8, 6))
        # plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_labels, s=5, alpha=0.7)
        # plt.xlabel('UMAP Dimension 1')
        # plt.ylabel('UMAP Dimension 2')
        # plt.title('UMAP Projection of Latent Representations')
        # plt.savefig(os, dpi=300, bbox_inches='tight')
        #

        opt_ctr = tf.keras.optimizers.Adam(learning_rate=0.001)
        opt_cls = tf.keras.optimizers.Adam(learning_rate=0.001)
        opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
        model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

        # Callbacks
        # num_training_samples = len(ds_train)
        # steps = EPOCHS * (num_training_samples // BATCH_SIZE)
        # lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03,
        #                                                             decay_steps=steps)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                                                    patience=5, min_lr=0.001)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        save_dir = os.path.join('saved_models')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, f'model_')
        save_model_callback = SaveModelCallback(model, save_path, save_best_only=True)
        save_best_valacc = SaveBestValAcc(save_best=False, filepath=os.path.join("logs", 'history', f'{dataset_name}_train{ds_train_sub}_test{ds_test_sub}.pkl'))

        save_umap_callback = SaveUMap(model, os.path.join('logs', 'figs'), ds_test, sub_idx= ds_test_sub[0] , idx=i)
        # End Callbacks

        history = model.fit(
                    ds_train,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_data=(ds_test),
                    callbacks=[tensorboard_callback, save_model_callback, save_best_valacc, save_umap_callback])

        print(All_val_acc)

        with open(os.path.join("logs", 'history', 'val_acc_sbj.pkl'), "wb") as pickle_file:
            pickle.dump(All_val_acc, pickle_file)
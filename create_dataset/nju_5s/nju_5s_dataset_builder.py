"""nju_5s dataset."""

import tensorflow_datasets as tfds
import dataclasses
import os
from typing import Tuple
import numpy as np
import h5py
import scipy.signal as signal
import scipy.io as sio
import librosa
import pandas as pd
from scipy.signal import remez
from scipy.signal import filtfilt, decimate
from mne.decoding import CSP

SR_EEG = 64
#SR_AUDIO = 16000
SR_DS_AUDIO=64
#TOTAL_TIME=50
T_SPAN = 5
N_CH = 32
CSP_DIM=32
UNPROCESSED = True

OV=0.5

@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    eeg_raw: Tuple[int, int] = (SR_EEG * T_SPAN, N_CH)
    eeg_csp:Tuple[int, int] = (CSP_DIM, SR_EEG * T_SPAN)
    eeg_csp_avg: Tuple[int,] = (CSP_DIM,)
    audio_l_ds: Tuple[int,] = (SR_EEG * T_SPAN,)
    audio_r_ds: Tuple[int,] = (SR_EEG * T_SPAN,)
    att_lr: Tuple[int,] = (None,)
    subject: Tuple[int,] = (None,)
    t_span: Tuple[int,] = (None,)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for nju_5s dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
        BuilderConfigEEG(name='s02', description='5 second splits, subject 02', subject=2, t_span=5),
        BuilderConfigEEG(name='s03', description='5 second splits, subject 03', subject=3, t_span=5),
        BuilderConfigEEG(name='s04', description='5 second splits, subject 04', subject=4, t_span=5),
        BuilderConfigEEG(name='s06', description='5 second splits, subject 06', subject=6, t_span=5),
        BuilderConfigEEG(name='s07', description='5 second splits, subject 07', subject=7, t_span=5),
        BuilderConfigEEG(name='s08', description='5 second splits, subject 08', subject=8, t_span=5),
        BuilderConfigEEG(name='s09', description='5 second splits, subject 09', subject=9, t_span=5),
        BuilderConfigEEG(name='s12', description='5 second splits, subject 12', subject=12, t_span=5),
        BuilderConfigEEG(name='s13', description='5 second splits, subject 13', subject=13, t_span=5),
        BuilderConfigEEG(name='s14', description='5 second splits, subject 14', subject=14, t_span=5),
        BuilderConfigEEG(name='s15', description='5 second splits, subject 15', subject=15, t_span=5),
        BuilderConfigEEG(name='s16', description='5 second splits, subject 16', subject=16, t_span=5),
        BuilderConfigEEG(name='s17', description='5 second splits, subject 17', subject=17, t_span=5),
        BuilderConfigEEG(name='s18', description='5 second splits, subject 18', subject=18, t_span=5),
        BuilderConfigEEG(name='s19', description='5 second splits, subject 19', subject=19, t_span=5),
        BuilderConfigEEG(name='s21', description='5 second splits, subject 21', subject=21, t_span=5),
        BuilderConfigEEG(name='s22', description='5 second splits, subject 22', subject=22, t_span=5),
        BuilderConfigEEG(name='s23', description='5 second splits, subject 23', subject=23, t_span=5),
        BuilderConfigEEG(name='s25', description='5 second splits, subject 25', subject=25, t_span=5),
        BuilderConfigEEG(name='s26', description='5 second splits, subject 26', subject=26, t_span=5),
        BuilderConfigEEG(name='s27', description='5 second splits, subject 27', subject=27, t_span=5),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(nju_5s): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'eeg_csp':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp),
            'audio_l_ds':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_l_ds),
            'audio_r_ds': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_r_ds),
            'att_lr': tfds.features.ClassLabel(names=['left', 'right']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    return {
            'train': self._generate_examples('train'),
        }

  def _generate_examples(self, path):
    idx_all = -1
    subject = self._builder_config.subject
    T_SPAN = self._builder_config.t_span
    STEP_SIZE_EEG = int(SR_EEG * T_SPAN * OV)
    STEP_SIZE_SDAUDIO = int(SR_DS_AUDIO * T_SPAN * OV)
    base_path ="/fs/ess/PAS2301/alialavi/datasets/nju/processed_data"

    df_labels = pd.read_csv('/fs/ess/PAS2301/alialavi/datasets/nju/labels/labels.csv', dtype=str)
    # labels = find_labels(df_labels, subject)

    eeg_ds_all_list=[]
    audio_l_ds_all_list=[]
    audio_r_ds_all_list=[]
    labels_all_list=[]

    with h5py.File(os.path.join(base_path, f"S{subject:02}_data_preproc.mat"), 'r') as mat_data:
      n_exp = len(mat_data['data'])
      ref =mat_data['data'][1,0]
      eeg = np.asarray(mat_data[ref]['eeg'][()])
      ref_label = mat_data[ref]['event/eeg/value'][()][1,0]
      ref_label=  mat_data[ref_label][()][0,0]
      
      fsample_eeg =64
      fsample_audio =64
      idx_cz=0
      for idx_exp in range(n_exp):
        ref =mat_data['data'][idx_exp,0]
        eeg_ds = np.asarray(mat_data[ref]['eeg'][()])
        audio_l_ds = np.asarray(mat_data[mat_data[ref]['wavA'][()][0,0]][()])[0,:]
        audio_r_ds = np.asarray(mat_data[mat_data[ref]['wavB'][()][0,0]][()])[0,:]
        label = mat_data[ref]['event/label'][()][0,0]
        # eeg_ds = eeg_ds - np.tile(eeg_ds[idx_cz,:], (32,1))  

        total_time = eeg_ds.shape[1]/SR_EEG
        assert(total_time==len(audio_r_ds)/SR_DS_AUDIO), f"Audio and EEG length don't match! eeg shape:{eeg_ds.shape}, audio shape:{audio_r_ds.shape}"

        n_sects = int((total_time - (1 - OV) * T_SPAN) // (OV*T_SPAN))
        for idx_sect in range(n_sects):
          begin_idx_eeg = idx_sect*(STEP_SIZE_EEG)
          end_idx_eeg = begin_idx_eeg + T_SPAN*SR_EEG
          begin_idx_audio = idx_sect*STEP_SIZE_SDAUDIO
          end_idx_audio = begin_idx_audio + T_SPAN*SR_DS_AUDIO

          eeg_ds_seg = eeg_ds[:,begin_idx_eeg:end_idx_eeg]
          audio_l_ds_seg = audio_l_ds[begin_idx_audio:end_idx_audio]
          audio_r_ds_seg = audio_r_ds[begin_idx_audio:end_idx_audio]

          eeg_ds_all_list.append(eeg_ds_seg)
          audio_l_ds_all_list.append(audio_l_ds_seg)
          audio_r_ds_all_list.append(audio_r_ds_seg)
          labels_all_list.append(label)

      eeg_ds_all = np.stack(eeg_ds_all_list, axis=0)
      audio_l_ds_all = np.stack(audio_l_ds_all_list, axis=0)
      audio_r_ds_all = np.stack(audio_r_ds_all_list, axis=0)
      labels_all = np.stack(labels_all_list, axis=0)

      csp = CSP(n_components=CSP_DIM, reg=None, log=True, norm_trace=False)
      # eeg_all_csp_avg = csp.fit_transform(eeg_ds_all.astype(np.float64), labels_all)
      csp.transform_into = 'csp_space'
      # noise = np.random.normal(loc=0, scale=0.001, size=eeg_ds_all.shape)
      # eeg_ds_all += noise
      eeg_all_csp = csp.fit_transform(eeg_ds_all.astype(np.float64), labels_all)

      n_data_points = eeg_ds_all.shape[0]

      for idx_data_point in range(n_data_points):
        curr_eeg_csp = eeg_all_csp[idx_data_point,...]
        curr_audio_l_ds = audio_l_ds_all[idx_data_point,...]
        curr_audio_r_ds = audio_r_ds_all[idx_data_point,...]
        curr_label = labels_all[idx_data_point]

        idx_all += 1
        yield idx_all, {
          'eeg_csp':curr_eeg_csp.astype(np.float32),
          'audio_l_ds': curr_audio_l_ds.astype(np.float32),
          'audio_r_ds': curr_audio_r_ds.astype(np.float32),
          'att_lr': curr_label,
        }

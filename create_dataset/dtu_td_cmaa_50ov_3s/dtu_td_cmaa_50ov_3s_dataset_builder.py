"""dtu_td_cmaa dataset."""

import tensorflow_datasets as tfds
import numpy as np
import scipy.io
import pandas as pd
import librosa
import dataclasses
import os
from typing import Tuple
from mne.decoding import CSP

SR_EEG = 64
SR_AUDIO = 16000
SR_DS_AUDIO=64
TOTAL_TIME=50
T_SPAN = 3
N_CH = 66
CSP_DIM=64
UNPROCESSED = True

OV=0.5



@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    eeg_raw: Tuple[int, int] = (int(SR_EEG * T_SPAN), N_CH)
    eeg_csp:Tuple[int, int] = (CSP_DIM, int(SR_EEG * T_SPAN))
    eeg_csp_avg: Tuple[int,] = (CSP_DIM,)
    audio_m_ds: Tuple[int,] = (int(SR_EEG * T_SPAN),)
    audio_f_ds: Tuple[int,] = (int(SR_EEG * T_SPAN),)
    audio_m_raw: Tuple[int,] = (int(SR_AUDIO * T_SPAN),)
    audio_f_raw: Tuple[int,] = (int(SR_AUDIO * T_SPAN),)
    att_gender: Tuple[int,] = (None,)
    att_lr: Tuple[int,] = (None,)
    subject: Tuple[int,] = (None,)
    t_span: Tuple[int,] = T_SPAN


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for eeg_den_single_latent dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        BuilderConfigEEG(name='sub1', description='5 second splits, subject 1', subject=1,),
        BuilderConfigEEG(name='sub2', description='5 second splits, subject 2', subject=2,),
        BuilderConfigEEG(name='sub3', description='5 second splits, subject 3', subject=3,),
        BuilderConfigEEG(name='sub4', description='5 second splits, subject 4', subject=4,),
        BuilderConfigEEG(name='sub5', description='5 second splits, subject 5', subject=5,),
        BuilderConfigEEG(name='sub6', description='5 second splits, subject 6', subject=6,),
        BuilderConfigEEG(name='sub7', description='5 second splits, subject 7', subject=7,),
        BuilderConfigEEG(name='sub8', description='5 second splits, subject 8', subject=8,),
        BuilderConfigEEG(name='sub9', description='5 second splits, subject 9', subject=9,),
        BuilderConfigEEG(name='sub10', description='5 second splits, subject 10', subject=10),
        BuilderConfigEEG(name='sub11', description='5 second splits, subject 11', subject=11),
        BuilderConfigEEG(name='sub12', description='5 second splits, subject 12', subject=12),
        BuilderConfigEEG(name='sub13', description='5 second splits, subject 13', subject=13),
        BuilderConfigEEG(name='sub14', description='5 second splits, subject 14', subject=14),
        BuilderConfigEEG(name='sub15', description='5 second splits, subject 15', subject=15),
        BuilderConfigEEG(name='sub16', description='5 second splits, subject 16', subject=16),
        BuilderConfigEEG(name='sub17', description='5 second splits, subject 17', subject=17),
        BuilderConfigEEG(name='sub18', description='5 second splits, subject 18', subject=18),
        BuilderConfigEEG(name='all', description='5 second splits, all subjects', subject=0),
        BuilderConfigEEG(name='all_e1', description='5 second splits, all subjects except 1', subject=-1),
        BuilderConfigEEG(name='all_e2', description='5 second splits, all subjects except 2', subject=-2),
        BuilderConfigEEG(name='all_e3', description='5 second splits, all subjects except 3', subject=-3),
        BuilderConfigEEG(name='all_e4', description='5 second splits, all subjects except 4', subject=-4),
        BuilderConfigEEG(name='all_e5', description='5 second splits, all subjects except 5', subject=-5),
        BuilderConfigEEG(name='all_e6', description='5 second splits, all subjects except 6', subject=-6),
        BuilderConfigEEG(name='all_e7', description='5 second splits, all subjects except 7', subject=-7),
        BuilderConfigEEG(name='all_e8', description='5 second splits, all subjects except 8', subject=-8),
        BuilderConfigEEG(name='all_e9', description='5 second splits, all subjects except 9', subject=-9),
        BuilderConfigEEG(name='all_e10', description='5 second splits, all subjects except 10', subject=-10),
        BuilderConfigEEG(name='all_e11', description='5 second splits, all subjects except 11', subject=-11),
        BuilderConfigEEG(name='all_e12', description='5 second splits, all subjects except 12', subject=-12),
        BuilderConfigEEG(name='all_e13', description='5 second splits, all subjects except 13', subject=-13),
        BuilderConfigEEG(name='all_e14', description='5 second splits, all subjects except 14', subject=-14),
        BuilderConfigEEG(name='all_e15', description='5 second splits, all subjects except 15', subject=-15),
        BuilderConfigEEG(name='all_e16', description='5 second splits, all subjects except 16', subject=-16),
        BuilderConfigEEG(name='all_e17', description='5 second splits, all subjects except 17', subject=-17),
        BuilderConfigEEG(name='all_e18', description='5 second splits, all subjects except 18', subject=-18),

    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(eeg_den_single_latent): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'eeg': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_raw),
                'eeg_csp':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp),
                'eeg_csp_avg': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp_avg),
                'audio_m_ds':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_m_ds),
                'audio_f_ds': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_f_ds),
                'audio_m_raw': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_m_raw),
                'audio_f_raw': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_f_raw),
                'att_gender': tfds.features.ClassLabel(names=['male', 'female']),
                'att_lr': tfds.features.ClassLabel(names=['left', 'right']),
                'aucostic': tfds.features.ClassLabel(names=['cond0', 'cond1', 'cond2']),
            }),
            supervised_keys=(None),
            homepage='https://zenodo.org/record/3377911',
            disable_shuffling=True
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(eeg_den_single_latent): Downloads the data and defines the splits

        # TODO(eeg_den_single_latent): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples('train'),
            'test': self._generate_examples('test'),
        }

    def _generate_examples(self,split):

        idx_all = -1
        AUDIO_MAXLEN = 246000
        LABEL_MAXLEN = 256
        BATCH_SIZE = 1

        subject = self._builder_config.subject
        T_SPAN = self._builder_config.t_span
        STEP_SIZE_EEG = int(SR_EEG * T_SPAN * OV)
        STEP_SIZE_AUDIO = int(SR_AUDIO * T_SPAN * OV)
        STEP_SIZE_SDAUDIO = int(SR_DS_AUDIO * T_SPAN * OV)
        base_path ="/users/PAS2301/alialavi/datasets/dtu/eeg_den/eeg_den"

        n_sects = int(50 // T_SPAN)
        fn_chan = os.path.join(base_path,'data/chan.mat' )
        f_chan_names = scipy.io.loadmat(fn_chan)
        ch_names = []
        for i in range(64):
            ch_names.append(f_chan_names['x'][i, 0][0])

        if subject >0:
            subjects = [subject]
        elif subject==0:
            subjects = list(range(1,19))
        else:
            if split=='train':
                subject= -subject
                subjects = list(range(1, 19))
                subjects.remove(subject)
            elif split=='test':
                subjects = [-subject]

        for subject in subjects:
            print(f'Subject: { subject}')
            if UNPROCESSED:
                fn_eeg = os.path.join(base_path, f'data/DATA_preproc/S{subject}_data_preproc.mat')
                f_eeg = scipy.io.loadmat(fn_eeg)

                fsample_eeg = f_eeg['data']['fsample'][0][0][0][0][0][0][0]
                n_data = len(f_eeg['data']['eeg'][0, 0][0])
                eeg = f_eeg['data']['eeg'][0, 0]
            else:
                raise ValueError('Processed data unspecified.')

            fn_expinfo = os.path.join(base_path, f'data/EEG/s{subject}_expinfo.txt')
            expinfo = pd.read_csv(fn_expinfo, delimiter=' ')
            expinfo = expinfo[expinfo['n_speakers'] == 2].reset_index()

            eeg_shape = np.asarray(eeg[0,:]).shape
            #eeg_all shape: samples, channels, times
            sh = eeg[0, 1].shape
            eeg_all = np.zeros((eeg.shape[1], *sh), dtype=np.float32)
            for i in range(eeg[0, :].shape[0]):
                eeg_all[i, ...] = eeg[0, i]
            eeg_all = eeg_all.transpose(0, 2, 1) #sample, channel, time
            # print(f'eeg_all shape: {eeg_all.shape}')
            n_sects = int((TOTAL_TIME - (1 - OV) * T_SPAN) // (OV*T_SPAN))

            eeg_all_reshaped = np.zeros((n_data*n_sects,eeg_all.shape[1],int(T_SPAN*SR_EEG)), dtype=np.float32)
            for curr_data in range(n_data):
                for curr_sect in range(n_sects):
                    curr_ind = curr_data*n_sects+curr_sect
                    eeg_all_reshaped[curr_ind,:,:] = eeg_all[curr_data, :, curr_sect*STEP_SIZE_EEG: curr_sect*STEP_SIZE_EEG+int(T_SPAN*SR_EEG)]

            # tmp=eeg_all.reshape((eeg_all.shape[0] * 10, eeg_all.shape[1], eeg_all.shape[2] // 10))

            event_mf_all = np.asarray(expinfo['attend_mf']).reshape(-1) - 1
            event_lr_all = np.asarray(expinfo['attend_lr']).reshape(-1) - 1
            event_acoustic_condition_all = np.asarray(expinfo['acoustic_condition']).reshape(-1) - 1
            events_mf_all_reshaped = np.repeat(event_mf_all, n_sects)
            event_lr_all_reshaped = np.repeat(event_lr_all, n_sects)
            event_acoustic_condition_all_reshaped = np.repeat(event_acoustic_condition_all, n_sects)

            wavfile_m_all =  np.asarray(expinfo['wavfile_male']).reshape(-1)
            wavfile_f_all = np.asarray(expinfo['wavfile_female']).reshape(-1)
            audio_m_ds_all=np.zeros((n_data, f_eeg['data']['wavA'][0,0][0,1][:,0].shape[0])).astype(np.float32)
            audio_f_ds_all=np.zeros((n_data, f_eeg['data']['wavA'][0,0][0,1][:,0].shape[0])).astype(np.float32)
            audio_m_raw_all = np.zeros((n_data, SR_AUDIO * 50), dtype=np.float32)
            audio_f_raw_all = np.zeros((n_data, SR_AUDIO * 50), dtype=np.float32)
            for idx_data in range(n_data):
                audio_m_ds = f_eeg['data']['wavA'][0, 0][0, idx_data][:, 0]
                audio_f_ds = f_eeg['data']['wavB'][0, 0][0, idx_data][:, 0]
                audio_m_ds_all[idx_data,...]=audio_m_ds
                audio_f_ds_all[idx_data,...]=audio_f_ds
                audio_m_raw_all[idx_data, :], _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO',   wavfile_m_all[idx_data]), sr=SR_AUDIO)
                audio_f_raw_all[idx_data, :], _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO', wavfile_f_all[idx_data]), sr=SR_AUDIO)

            audio_m_ds_all_reshaped = np.zeros((n_data*n_sects,int(T_SPAN*SR_DS_AUDIO)))
            audio_f_ds_all_reshaped = np.zeros((n_data * n_sects, int(T_SPAN * SR_DS_AUDIO)))
            audio_m_raw_all_reshaped = np.zeros((n_data * n_sects, int(T_SPAN * SR_AUDIO)))
            audio_f_raw_all_reshaped = np.zeros((n_data * n_sects, int(T_SPAN * SR_AUDIO)))
            for curr_data in range(n_data):
                for curr_sect in range(n_sects):
                    curr_ind = curr_data*n_sects+curr_sect
                    audio_m_ds_all_reshaped[curr_ind,...]=audio_m_ds_all[curr_data,curr_sect*STEP_SIZE_SDAUDIO:curr_sect*STEP_SIZE_SDAUDIO+int(T_SPAN*SR_DS_AUDIO)]
                    audio_f_ds_all_reshaped[curr_ind,...]=audio_f_ds_all[curr_data,curr_sect*STEP_SIZE_SDAUDIO:curr_sect*STEP_SIZE_SDAUDIO+int(T_SPAN*SR_DS_AUDIO)]
                    audio_m_raw_all_reshaped[curr_ind,...]=audio_m_raw_all[curr_data,curr_sect*STEP_SIZE_AUDIO:curr_sect*STEP_SIZE_AUDIO+int(T_SPAN*SR_AUDIO)]
                    audio_f_raw_all_reshaped[curr_ind,...]=audio_f_raw_all[curr_data,curr_sect*STEP_SIZE_AUDIO:curr_sect*STEP_SIZE_AUDIO+int(T_SPAN*SR_AUDIO)]

            # if subject == 2:
            #     eeg_all_reshaped+=np.random.normal(0, 0.01)
            if subject == 10 or subject==11 or subject ==2 or subject == 5 or subject==8:
                eeg_all_reshaped += np.random.normal(0, 0.001)
            if self._builder_config.subject<0:
                eeg_all_reshaped += np.random.normal(0, 0.000001)
            eeg_all_reshaped += np.random.normal(0, 0.000001)

            csp = CSP(n_components=CSP_DIM, reg=None, log=True, norm_trace=False)
            eeg_all_csp_avg_reshaped = csp.fit_transform(eeg_all_reshaped.astype(np.float64), events_mf_all_reshaped)
            csp.transform_into = 'csp_space'
            eeg_all_csp_reshaped = csp.fit_transform(eeg_all_reshaped.astype(np.float64), events_mf_all_reshaped)
            print(f'csp shape {eeg_all_csp_reshaped.shape}')
            n_data_points = n_sects*n_data
            assert n_data_points==eeg_all_reshaped.shape[0]
            assert events_mf_all_reshaped.shape[0]==eeg_all_reshaped.shape[0]

            arr = np.array(np.arange(n_data_points))
            # np.random.shuffle(arr)
            #print(arr)
            for data_point in arr:
                eeg_curr=eeg_all_reshaped[data_point,...].transpose(1,0).astype(np.float32)
                eeg_transformed_curr = eeg_all_csp_reshaped[data_point,...].astype(np.float32)
                eeg_transformed_avg_curr = eeg_all_csp_avg_reshaped[data_point, ...].astype(np.float32)
                audio_male_curr=audio_m_raw_all_reshaped[data_point,...].astype(np.float32)
                audio_m_ds_curr=audio_m_ds_all_reshaped[data_point,...].astype(np.float32)

                audio_female_curr=audio_f_raw_all_reshaped[data_point,...].astype(np.float32)
                audio_f_ds_curr=audio_f_ds_all_reshaped[data_point,...].astype(np.float32)

                event_mf=events_mf_all_reshaped[data_point,...]
                event_lr=event_lr_all_reshaped[data_point,...]
                event_acoustic_condition=event_acoustic_condition_all_reshaped[data_point,...]

                idx_all += 1
                yield idx_all, {
                    'eeg': eeg_curr,
                    'eeg_csp':eeg_transformed_curr,
                    'eeg_csp_avg': eeg_transformed_avg_curr,
                    'audio_m_raw': audio_male_curr,
                    'audio_m_ds': audio_m_ds_curr,
                    'audio_f_ds': audio_f_ds_curr,
                    'audio_f_raw': audio_female_curr,
                    'att_gender': event_mf,
                    'att_lr': event_lr,
                    'aucostic': event_acoustic_condition
                }


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
T_SPAN = 30
N_CH = 64
CSP_DIM=64
UNPROCESSED = True

OV=0.5



@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    eeg_raw: Tuple[int, int] = (SR_EEG * T_SPAN, N_CH)
    eeg_csp:Tuple[int, int] = (CSP_DIM, SR_EEG * T_SPAN)
    eeg_csp_avg: Tuple[int,] = (CSP_DIM,)
    audio_m_ds: Tuple[int,] = (SR_EEG * T_SPAN,)
    audio_f_ds: Tuple[int,] = (SR_EEG * T_SPAN,)
    audio_m_raw: Tuple[int,] = (SR_AUDIO * T_SPAN,)
    audio_f_raw: Tuple[int,] = (SR_AUDIO * T_SPAN,)
    att_gender: Tuple[int,] = (None,)
    att_lr: Tuple[int,] = (None,)
    subject: Tuple[int,] = (None,)
    t_span: Tuple[int,] = T_SPAN


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for eeg_kuleuven dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        BuilderConfigEEG(name='sub1', description='30 second splits, subject 1', subject=1,),
        BuilderConfigEEG(name='sub2', description='30 second splits, subject 2', subject=2,),
        BuilderConfigEEG(name='sub3', description='30 second splits, subject 3', subject=3,),
        BuilderConfigEEG(name='sub4', description='30 second splits, subject 4', subject=4,),
        BuilderConfigEEG(name='sub5', description='30 second splits, subject 5', subject=5,),
        BuilderConfigEEG(name='sub6', description='30 second splits, subject 6', subject=6,),
        BuilderConfigEEG(name='sub7', description='30 second splits, subject 7', subject=7,),
        BuilderConfigEEG(name='sub8', description='30 second splits, subject 8', subject=8,),
        BuilderConfigEEG(name='sub9', description='30 second splits, subject 9', subject=9,),
        BuilderConfigEEG(name='sub10', description='30 second splits, subject 10', subject=10),
        BuilderConfigEEG(name='sub11', description='30 second splits, subject 11', subject=11),
        BuilderConfigEEG(name='sub12', description='30 second splits, subject 12', subject=12),
        BuilderConfigEEG(name='sub13', description='30 second splits, subject 13', subject=13),
        BuilderConfigEEG(name='sub14', description='30 second splits, subject 14', subject=14),
        BuilderConfigEEG(name='sub15', description='30 second splits, subject 15', subject=15),
        BuilderConfigEEG(name='sub16', description='30 second splits, subject 16', subject=16),
        BuilderConfigEEG(name='all', description='30 second splits, all subjects', subject=0),
        BuilderConfigEEG(name='all_e1', description='30 second splits, all subjects except 1', subject=-1),
        BuilderConfigEEG(name='all_e2', description='30 second splits, all subjects except 2', subject=-2),
        BuilderConfigEEG(name='all_e3', description='30 second splits, all subjects except 3', subject=-3),
        BuilderConfigEEG(name='all_e4', description='30 second splits, all subjects except 4', subject=-4),
        BuilderConfigEEG(name='all_e5', description='30 second splits, all subjects except 5', subject=-5),
        BuilderConfigEEG(name='all_e6', description='30 second splits, all subjects except 6', subject=-6),
        BuilderConfigEEG(name='all_e7', description='30 second splits, all subjects except 7', subject=-7),
        BuilderConfigEEG(name='all_e8', description='30 second splits, all subjects except 8', subject=-8),
        BuilderConfigEEG(name='all_e9', description='30 second splits, all subjects except 9', subject=-9),
        BuilderConfigEEG(name='all_e10', description='30 second splits, all subjects except 10', subject=-10),
        BuilderConfigEEG(name='all_e11', description='30 second splits, all subjects except 11', subject=-11),
        BuilderConfigEEG(name='all_e12', description='30 second splits, all subjects except 12', subject=-12),
        BuilderConfigEEG(name='all_e13', description='30 second splits, all subjects except 13', subject=-13),
        BuilderConfigEEG(name='all_e14', description='30 second splits, all subjects except 14', subject=-14),
        BuilderConfigEEG(name='all_e15', description='30 second splits, all subjects except 15', subject=-15),
        BuilderConfigEEG(name='all_e16', description='30 second splits, all subjects except 16', subject=-16),

    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(eeg_den_single_latent): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'eeg': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_raw),
                'eeg_csp':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp),
                'eeg_csp_lr':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp),
                'eeg_csp_avg': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg_csp_avg),
                'audio_0_ds':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_m_ds),
                'audio_1_ds': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_f_ds),
                'att_src': tfds.features.ClassLabel(names=['track_0', 'track_1']),
                'att_lr': tfds.features.ClassLabel(names=['left', 'right']),
                'aucostic': tfds.features.ClassLabel(names=['htrf', 'dry']),
            }),
            supervised_keys=(None),
            homepage='https://zenodo.org/records/1199011',
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
        base_dir ="/fs/ess/PAS2301/alialavi/datasets/kuleuven"

        n_sects = int(50 // T_SPAN)

        if subject >0:
            # if split=='train':
            subjects = [subject]
            # else:
            #     subjects = []
        elif subject==0:
            # if split=='train':
            subjects = list(range(1,17))
            # else:
            #     subjects= []
        else:
            if split=='train':
                subject= -subject
                subjects = list(range(1, 17))
                subjects.remove(subject)
            elif split=='test':
                subjects = [-subject]

        for subject in subjects:
            print(f'Subject {subject}')
            fn_eeg = os.path.join(base_dir, f'preprocessed_data/S{subject}.mat')
            f_eeg = scipy.io.loadmat(fn_eeg)

            n_trials = f_eeg['preproc_trials'].shape[1]

            # lr_all = np.array([])
            # att_track_all = np.array([])
            # condition_all = np.array([])
            # eeg_all = np.array([])
            # audio_0_all=np.array([])
            # audio_1_all=np.array([])

            lr_all = []
            att_track_all = []
            condition_all = []
            eeg_all = []
            audio_0_all=[]
            audio_1_all=[]

            for idx_trial in range(n_trials):
                # if idx_trial==5:
                #     break
                trial = f_eeg['preproc_trials'][0, idx_trial]
                lr = int(trial[0, 0][3][0] == 'L')
                att_track = trial[0, 0][8][0][0] - 1
                eeg = trial[0, 0][0][0, 0][1]

                sr_eeg = trial[0, 0][1][0, 0][10][0, 0]
                condition = int(trial[0, 0][5][0]=='hrtf')

                track_0 = trial[0, 0][4][0, 0][0]
                track_1 = trial[0, 0][4][1, 0][0]
                track_0 = os.path.join(base_dir, 'stimuli', track_0)
                track_0, _ = librosa.load(track_0, sr=SR_AUDIO)

                track_1 = os.path.join(base_dir, 'stimuli', track_1)
                track_1, _ = librosa.load(track_1, sr=SR_AUDIO)

                track_0_ds = trial[0,0][11][0,0][0][:,0,0]
                track_1_ds = trial[0,0][11][0,0][0][:,0,1]

                TOTAL_TIME =eeg.shape[0]/sr_eeg

                n_sects = int((TOTAL_TIME - (1 - OV) * T_SPAN) // (OV * T_SPAN))
                n_channels = eeg.shape[1]

                eeg_all_curr = np.zeros((n_sects, n_channels, SR_EEG*T_SPAN), dtype=np.float32)
                audio_0_all_curr = np.zeros((n_sects, SR_DS_AUDIO*T_SPAN), dtype=np.float32)
                audio_1_all_curr = np.zeros((n_sects, SR_DS_AUDIO * T_SPAN), dtype=np.float32)

                lr = np.ones(n_sects, dtype=int)*lr
                att_track = np.ones(n_sects, dtype=int)*att_track
                condition = np.ones(n_sects, dtype=int)*condition
                for sect_idx in range(n_sects):
                    # print(f'subject:{subject}  indext_trial:{idx_trial}/{n_trials} sectid:{sect_idx}/{n_sects}')
                    # if sect_idx==3:
                    #     break
                    # print(f'eeg shape selected: {eeg[sect_idx*STEP_SIZE_EEG:sect_idx*STEP_SIZE_EEG+(SR_EEG*T_SPAN),:].transpose(1,0).shape}')
                    eeg_all_curr[sect_idx,:,:] = eeg[sect_idx*STEP_SIZE_EEG:sect_idx*STEP_SIZE_EEG+(SR_EEG*T_SPAN),:].transpose(1,0) #sample, channel, time
                    # print(f'eeg shape:{eeg_all_curr.shape}')
                    audio_0_all_curr[sect_idx, :] = track_0_ds[sect_idx*STEP_SIZE_SDAUDIO:sect_idx*STEP_SIZE_SDAUDIO+(SR_DS_AUDIO*T_SPAN)]
                    audio_1_all_curr[sect_idx, :] = track_1_ds[sect_idx*STEP_SIZE_SDAUDIO:sect_idx*STEP_SIZE_SDAUDIO+(SR_DS_AUDIO*T_SPAN)]
                    # print(f'subject:{subject}  indext_trial:{idx_trial}/{n_trials} sectid:{sect_idx}/{n_sects} eegshape:{eeg_all_curr.shape}')
                    # if eeg_all.shape[0]==0:
                    #     eeg_all = eeg_all_curr
                    #     audio_0_all = audio_0_all_curr
                    #     audio_1_all = audio_1_all_curr
                    #     lr_all = lr
                    #     att_track_all = att_track
                    #     condition_all = condition
                    # else:
                    # eeg_all=np.append(eeg_all,eeg_all_curr,axis=0)
                    # audio_0_all=np.append(audio_0_all,audio_0_all_curr,axis=0)
                    # audio_1_all=np.append(audio_1_all,audio_1_all_curr,axis=0)
                    # lr_all = np.append(lr_all,lr,axis=0)
                    # att_track_all = np.append(att_track_all,att_track,axis=0)
                    # condition_all = np.append(condition_all,condition, axis=0)

                eeg_all.append(eeg_all_curr)
                audio_0_all.append(audio_0_all_curr)
                audio_1_all.append(audio_1_all_curr)
                lr_all.append(lr)
                att_track_all.append(att_track)
                condition_all.append(condition)
            # print(f'len eeg_all:{len(eeg_all)}')
            # for i in range(len(eeg_all)):
            #     print(f'element {i} length {len(eeg_all[i])}')
            eeg_all=np.concatenate(eeg_all,axis=0)
            # print(f'eeg shape:{eeg_all.shape}')
            audio_0_all=np.concatenate(audio_0_all,axis=0)
            # print(f'audio_0_all shape:{audio_0_all.shape}')
            audio_1_all=np.concatenate(audio_1_all,axis=0)
            # print(f'audio_1_all shape:{audio_1_all.shape}')
            lr_all=np.concatenate(lr_all,axis=0)
            att_track_all=np.concatenate(att_track_all,axis=0)
            condition_all=np.concatenate(condition_all,axis=0)
            # print('CSP processing.')
            csp = CSP(n_components=CSP_DIM, reg=None, log=True, norm_trace=False)
            eeg_all_csp_avg_reshaped = csp.fit_transform(eeg_all.astype(np.float64), att_track_all)
            csp.transform_into = 'csp_space'
            eeg_all_csp_reshaped = csp.fit_transform(eeg_all.astype(np.float64), att_track_all)
            eeg_all_csp_reshaped_lr = csp.fit_transform(eeg_all.astype(np.float64), lr_all)
            data_points = eeg_all.shape[0]

            for data_point_idx in range(data_points):
                # print(data_point_idx)
                eeg_curr = eeg_all[data_point_idx,...].transpose(1,0)
                eeg_transformed_curr =eeg_all_csp_reshaped[data_point_idx,...].astype(np.float32)
                eeg_transformed_avg_curr =eeg_all_csp_avg_reshaped[data_point_idx,...].astype(np.float32)
                eeg_csp_lr_curr = eeg_all_csp_reshaped_lr[data_point_idx,...].astype(np.float32)
                audio_0_ds_curr =audio_0_all[data_point_idx,...]
                audio_1_ds_curr =audio_1_all[data_point_idx,...]
                att_src_curr =att_track_all[data_point_idx]
                event_lr_curr =lr_all[data_point_idx]
                event_acoustic_condition_curr =condition_all [data_point_idx]

                idx_all+=1
                yield idx_all, {
                    'eeg': eeg_curr,
                    'eeg_csp': eeg_transformed_curr,
                    'eeg_csp_lr':eeg_csp_lr_curr,
                    'eeg_csp_avg': eeg_transformed_avg_curr,
                    'audio_0_ds': audio_0_ds_curr,
                    'audio_1_ds': audio_1_ds_curr,
                    'att_src': att_src_curr,
                    'att_lr': event_lr_curr,
                    'aucostic': event_acoustic_condition_curr
                }


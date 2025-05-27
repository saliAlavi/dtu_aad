"""ku_luven_tfds dataset."""

import tensorflow_datasets as tfds
import numpy as np
import scipy.io
import pandas as pd
import librosa
import dataclasses
import os
from typing import Tuple
from scipy import signal

SR = 16000
T_SPAN = 5
N_CH = 66
FS=64
UNPROCESSED = True


@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    eeg: Tuple[int, int] = (161,16, N_CH)
    audio_m_raw: Tuple[int,] = (SR * 5,)
    audio_f_raw: Tuple[int,] = (SR * 5,)
    att_gender: Tuple[int,] = (None,)
    att_lr: Tuple[int,] = (None,)
    subject: Tuple[int,] = (None,)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ku_luven_tfds dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = [
        BuilderConfigEEG(name='5s_sub1', description='5 second splits, subject 1', subject=1),
        BuilderConfigEEG(name='5s_sub2', description='5 second splits, subject 2', subject=2),
        BuilderConfigEEG(name='5s_sub3', description='5 second splits, subject 3', subject=3),
        BuilderConfigEEG(name='5s_sub4', description='5 second splits, subject 4', subject=4),
        BuilderConfigEEG(name='5s_sub5', description='5 second splits, subject 5', subject=5),
        BuilderConfigEEG(name='5s_sub6', description='5 second splits, subject 6', subject=6),
        BuilderConfigEEG(name='5s_sub7', description='5 second splits, subject 7', subject=7),
        BuilderConfigEEG(name='5s_sub8', description='5 second splits, subject 8', subject=8),
        BuilderConfigEEG(name='5s_sub9', description='5 second splits, subject 9', subject=9),
        BuilderConfigEEG(name='5s_sub10', description='5 second splits, subject 10', subject=10),
        BuilderConfigEEG(name='5s_sub11', description='5 second splits, subject 11', subject=11),
        BuilderConfigEEG(name='5s_sub12', description='5 second splits, subject 12', subject=12),
        BuilderConfigEEG(name='5s_sub13', description='5 second splits, subject 13', subject=13),
        BuilderConfigEEG(name='5s_sub14', description='5 second splits, subject 14', subject=14),
        BuilderConfigEEG(name='5s_sub15', description='5 second splits, subject 15', subject=15),
        BuilderConfigEEG(name='5s_sub16', description='5 second splits, subject 16', subject=16),
        BuilderConfigEEG(name='5s_sub17', description='5 second splits, subject 17', subject=17),
        BuilderConfigEEG(name='5s_sub18', description='5 second splits, subject 18', subject=18),
        BuilderConfigEEG(name='5s_all', description='5 second splits, all subjects', subject=-1),

    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(eeg_den_single_latent): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'eeg_f_real': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg),
                'eeg_f_imag': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.eeg),
                'audio_m_raw': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_m_raw),
                'audio_f_raw': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_f_raw),
                'att_gender': tfds.features.ClassLabel(names=['male', 'female']),
                'att_lr': tfds.features.ClassLabel(names=['left', 'right']),
                'aucostic': tfds.features.ClassLabel(names=['cond0', 'cond1', 'cond2']),
            }),
            supervised_keys=(None),
            homepage='https://zenodo.org/record/3377911',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(eeg_den_single_latent): Downloads the data and defines the splits

        # TODO(eeg_den_single_latent): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):

        idx_all = -1
        AUDIO_MAXLEN = 246000
        LABEL_MAXLEN = 256
        BATCH_SIZE = 1

        subject = self._builder_config.subject

        base_path ="/home/alialavi/datasets/eeg_den"

        fn_chan = os.path.join(base_path,'data/chan.mat' )
        f_chan_names = scipy.io.loadmat(fn_chan)
        ch_names = []
        for i in range(64):
            ch_names.append(f_chan_names['x'][i, 0][0])

        if subject > 0:
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

            for idx_data in range(n_data):
                eeg_exp = eeg[0, idx_data]
                event_lr = expinfo['attend_lr'][idx_data] - 1
                event_mf = expinfo['attend_mf'][idx_data] - 1
                event_acoustic_condition = expinfo['acoustic_condition'][idx_data] - 1
                wavfile_m = expinfo['wavfile_male'][idx_data]
                wavfile_f = expinfo['wavfile_female'][idx_data]

                audio_male, _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO', wavfile_m), sr=SR)
                audio_female, _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO', wavfile_f), sr=SR)

                n_sects = int(50 // T_SPAN)

                for i_sect in range(n_sects):
                    eeg_curr = np.asarray(eeg_exp[i_sect * fsample_eeg * T_SPAN:(i_sect + 1) * fsample_eeg * T_SPAN, :],
                                          dtype=np.float32)
                    audio_male_curr = audio_male[i_sect * SR * T_SPAN:(i_sect + 1) * SR * T_SPAN]
                    audio_female_curr = audio_female[i_sect * SR * T_SPAN:(i_sect + 1) * SR * T_SPAN]

                    eeg_tfds = np.zeros(self._builder_config.eeg, dtype=np.complex)
                    for ch in range(66):
                        eeg_tfds[:,:, ch] = librosa.stft(eeg_curr[:, ch], n_fft=int(30), hop_length=int(2)).T

                    idx_all += 1
                    yield idx_all, {
                        'eeg_f_real': (eeg_tfds.real).astype(np.float32),
                        'eeg_f_imag': (eeg_tfds.imag).astype(np.float32),
                        'audio_m_raw': audio_male_curr,
                        'audio_f_raw': audio_female_curr,
                        'att_gender': event_mf,
                        'att_lr': event_lr,
                        'aucostic': event_acoustic_condition
                    }
        else:
            for subject in range(1,19):
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

                for idx_data in range(n_data):
                    eeg_exp = eeg[0, idx_data]
                    event_lr = expinfo['attend_lr'][idx_data] - 1
                    event_mf = expinfo['attend_mf'][idx_data] - 1
                    event_acoustic_condition = expinfo['acoustic_condition'][idx_data] - 1
                    wavfile_m = expinfo['wavfile_male'][idx_data]
                    wavfile_f = expinfo['wavfile_female'][idx_data]

                    audio_male, _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO', wavfile_m), sr=SR)
                    audio_female, _ = librosa.load(os.path.join(base_path, 'data', 'AUDIO', wavfile_f), sr=SR)

                    n_sects = int(50 // T_SPAN)

                    for i_sect in range(n_sects):
                        eeg_curr = np.asarray(
                            eeg_exp[i_sect * fsample_eeg * T_SPAN:(i_sect + 1) * fsample_eeg * T_SPAN, :],
                            dtype=np.float32)
                        audio_male_curr = audio_male[i_sect * SR * T_SPAN:(i_sect + 1) * SR * T_SPAN]
                        audio_female_curr = audio_female[i_sect * SR * T_SPAN:(i_sect + 1) * SR * T_SPAN]

                        eeg_tfds = np.zeros(self._builder_config.eeg, dtype=np.complex)
                        for ch in range(66):
                            eeg_tfds[:, :, ch] = librosa.stft(eeg_curr[:, ch], n_fft=int(30), hop_length=int(2)).T

                        idx_all += 1
                        yield idx_all, {
                            'eeg_f_real': (eeg_tfds.real).astype(np.float32),
                            'eeg_f_imag': (eeg_tfds.imag).astype(np.float32),
                            'audio_m_raw': audio_male_curr,
                            'audio_f_raw': audio_female_curr,
                            'att_gender': event_mf,
                            'att_lr': event_lr,
                            'aucostic': event_acoustic_condition
                        }
"""kuleuven_td dataset."""

import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple
import scipy.io
import librosa
import dataclasses
import os
import tensorflow as tf


EEG_SR = 32
AUDIO_SR=8000
WL=5
N_CH=64

@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    data: Tuple[int, int] = (EEG_SR*WL, N_CH)
    date_f_real:Tuple[int, int] =  (151, 17,N_CH)
    data_f_imag: Tuple[int, int] = (151, 17,N_CH)
    audio_l: Tuple[int,] = (AUDIO_SR * WL,)
    audio_r: Tuple[int,] = (AUDIO_SR * WL ,)
    label_lr: Tuple[int,] = (None,)
    label_attsrc: Tuple[int,] = (None,)
    label_experiment: Tuple[int,] = (None,)
    label_condition: Tuple[int,] = (None,)
    subject: Tuple[int,] = (None,)


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kuleuven_td dataset."""

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
      BuilderConfigEEG(name='5s_all', description='5 second splits, all subjects', subject=-1),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(kuleuven_td): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'data': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.data),
            'data_f_real': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.date_f_real),
            'data_f_imag': tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.data_f_imag),
            'audio_l':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_l),
            'audio_r':tfds.features.Tensor(dtype=np.float32, shape=self._builder_config.audio_r),
            'label_lr': tfds.features.ClassLabel(names=['left', 'right']),
            'label_attsrc':tfds.features.ClassLabel(num_classes=2),
            'label_experiment':tfds.features.ClassLabel(num_classes=3),
            'label_condition':tfds.features.ClassLabel(num_classes=2),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('data', 'audio_l', 'audio_r'),  # Set to `None` to disable
        homepage='https://zenodo.org/record/3377911',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(kuleuven_td): Downloads the data and defines the splits
    # TODO(kuleuven_td): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(),
    }

  def _generate_examples(self):
    """Yields examples."""
    # TODO(kuleuven_td): Yields (key, example) tuples from the dataset
    idx_all = -1
    AUDIO_MAXLEN = 246000
    LABEL_MAXLEN = 256
    BATCH_SIZE = 1

    subject = self._builder_config.subject

    base_path ="/home/alialavi/datasets/KULeuven Dataset/preprocessed_data/processed"
    if subject>0:
       subjects = [subject]
    else:
       subjects = list(range(1,17))

    for subject in subjects:
      for idx_experiment in range(1,21):
          data_path = os.path.join(base_path, f'S{subject}_X{idx_experiment}.mat')
          data = scipy.io.loadmat(data_path)
          data = data['eeg_data'].astype(np.float32)

          label_path = os.path.join(base_path, f'S{subject}_L{idx_experiment}.mat')
          labels = scipy.io.loadmat(label_path)

          audio_src_0 = labels['labels'][0,0][2][0,0]
          audio_src_0 = labels['labels'][0,0][2][0,1]
          label_lr = 0 if labels['labels'][0,0][0][0]=='R' else 1
          label_atttrack = labels['labels'][0,0][1][0][0]-1
          label_stimuli =labels['labels'][0,0][2][0]
          label_condition = 0 if (labels['labels'][0,0][3]=='hrtf')[0] else 1
          label_experiment =labels['labels'][0,0][4][0][0]-1

          base_path_audio = "/home/alialavi/datasets/KULeuven Dataset/stimuli/stimuli"
          if label_lr==0:
            audio_0, _ = librosa.load(os.path.join(base_path_audio, label_stimuli[0][0]), sr=AUDIO_SR)
            audio_1, _ = librosa.load(os.path.join(base_path_audio, label_stimuli[1][0]), sr=AUDIO_SR)
          else:
            audio_1, _ = librosa.load(os.path.join(base_path_audio, label_stimuli[0][0]), sr=AUDIO_SR)
            audio_0, _ = librosa.load(os.path.join(base_path_audio, label_stimuli[1][0]), sr=AUDIO_SR)

          max_data_length = data.shape[0]
          max_audio_length = audio_0.shape[0]
          assert( audio_0.shape[0]==audio_1.shape[0])
          n_sections = max_data_length//(EEG_SR*WL)
          eeg_section_len = EEG_SR*WL
          audio_section_len = AUDIO_SR*WL
          # print(n_sections)
          # print(max_audio_length//audio_section_len)
          # assert(n_sections== (max_audio_length//audio_section_len) )

          for idx_data in range(n_sections):
            curr_data = data[idx_data*eeg_section_len: (idx_data+1)*eeg_section_len, :]
            curr_audio_0 = audio_0[idx_data*audio_section_len: (idx_data+1)*audio_section_len]
            curr_audio_1 = audio_1[idx_data*audio_section_len: (idx_data+1)*audio_section_len]

            frame_lenth = 10
            frame_step = 1
            fft_length = 32
            curr_data_real = np.zeros([*self._builder_config.date_f_real], dtype=np.float32)
            curr_data_imag = np.zeros([*self._builder_config.data_f_imag], dtype=np.float32)
            for ch in range(64):
                curr_data_ch = curr_data[:, ch]
                curr_data_ch = tf.signal.stft(curr_data_ch, frame_length=frame_lenth, frame_step=frame_step,
                                           fft_length=fft_length)
                curr_data_real[..., ch] = tf.math.real(curr_data_ch).numpy().astype(np.float32)
                curr_data_imag[..., ch] = tf.math.imag(curr_data_ch).numpy().astype(np.float32)

            idx_all += 1
            yield idx_all, {
                'data': curr_data,
                'data_f_real':curr_data_real,
                'data_f_imag':curr_data_imag,
                'audio_l': curr_audio_0,
                'audio_r': curr_audio_1,
                'label_lr': label_lr,
                'label_attsrc':label_atttrack,
                'label_experiment':label_experiment,
                'label_condition':label_condition,
            }

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow import keras
import math

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Dense(d_model)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x /= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class ConvUnetUnit(layers.Layer):
    def __init__(self, in_ch, out_ch, kernels = 3, dropout=0.1):
        super().__init__()
        self.cnn_in = layers.Conv1D(in_ch, kernels, strides = (1), padding='same', activation='relu')
        self.cnn_in_dropout = layers.Dropout(dropout)
        mid_layer_ch = (in_ch+out_ch)//4 if (in_ch+out_ch)//4>0 else 1
        self.cnn_mid = layers.Conv1D(mid_layer_ch, kernels, strides=(1), padding='same', activation='relu')
        self.cnn_mid_dropout = layers.Dropout(dropout)
        self.cnn_out = layers.Conv1D(out_ch, kernels, strides=(1), padding='same', activation='relu')

    def call(self, x):
        x = self.cnn_in(x)
        x = self.cnn_in_dropout(x)
        x = self.cnn_mid(x)
        x = self.cnn_mid_dropout(x)
        x = self.cnn_out(x)

        return x

class ConvSkip(layers.Layer):
    def __init__(self, ch, kernels = 3, dropout=0.1):
        super().__init__()
        self.cnn_in =  ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_mid = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_out = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)

    def call(self, x):
        # h =x
        x= self.cnn_in(x)
        x = self.cnn_mid(x) + x
        x = self.cnn_out(x) + x

        return x


class RNNUnit(layers.Layer):
    def __init__(self, rnn_units=32, in_dim = 32, out_dim=32, dropout= 0, return_sequence=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequence= return_sequence
        self.dense = layers.Dense(in_dim)
        # self.gru = tf.keras.layers.GRU(rnn_units,
        #                                return_sequences=True,
        #                                return_state=True)
        self.gru = layers.GRU(rnn_units, return_sequences=return_sequence, return_state=return_sequence, dropout=dropout)
        self.dense = layers.Dense(out_dim)

    def call(self,x, training=True):
        # states = self.gru.get_initial_state(x)
        # x, states = self.gru(x, initial_state=states, training=training)
        sequence, x = self.gru(x, training=training)
        x = self.dense(x)
        if self.return_sequence:
            return x, sequence
        else:
            return x


class SelfAttention(layers.Layer):
    def __init__(self, d_model = 50, num_heads=1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.query_emb = layers.Dense(d_model)
        self.key_emb = layers.Dense(d_model)
        self.value_emb = layers.Dense(d_model)

    def call(self, x):
        q = self.query_emb(x)
        k = self.key_emb(x)
        v = self.value_emb(x)
        h = self.mha(query = q, key= k, value= v)
        h = self.add([h,q])
        x = self.layernorm(h)

        return x

class CrossAttention(layers.Layer):
    def __init__(self, d_model=50, num_heads=1):
        super().__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.query_emb = layers.Dense(d_model)
        self.key_emb = layers.Dense(d_model)
        self.value_emb = layers.Dense(d_model)

    def call(self, x, contex):
        q = self.query_emb(x)
        k = self.key_emb(contex)
        v = self.value_emb(contex)
        h = self.mha(query = q, key= k, value= v)
        h = self.add([h,q])
        x = self.layernorm(h)

        return x


class EncoderImag(layers.Layer):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.pos_emb = PositionalEmbedding(d_model)
        self.convs = tf.keras.Sequential([
            ConvSkip(50),
            ConvSkip(25),
            ConvSkip(1),
        ])
        self.attns = tf.keras.Sequential([
            SelfAttention(d_model,num_heads=num_heads),
            SelfAttention(d_model,num_heads=num_heads),
            SelfAttention(d_model,num_heads=num_heads),
            SelfAttention(d_model,num_heads=num_heads),
        ])

    def call(self, x):
        x = self.convs(x)
        known_axes = [i for i, size in enumerate(x.shape) if size == 1]

        x = tf.squeeze(x, axis=known_axes)
        x = self.pos_emb(x)
        x = self.attns(x)

        return x

class EncoderReal(layers.Layer):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.pos_emb = PositionalEmbedding(d_model)
        self.convs = tf.keras.Sequential([
            ConvSkip(50),
            ConvSkip(25),
            ConvSkip(25),
            ConvSkip(25),
            ConvSkip(25),
            ConvSkip(25),
            ConvSkip(25),
            ConvSkip(25),
            # ConvSkip(1),
        ], name= 'conv_encreal')
        self.attns = tf.keras.Sequential([
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
            SelfAttention(num_heads=num_heads),
        ],name='attn_encreal')

    def call(self, x):
        x = self.convs(x)
        known_axes = [i for i, size in enumerate(x.shape) if size == 1]

        # x = tf.squeeze(x, axis=known_axes)
        x = self.pos_emb(x)
        # x += h
        x = self.attns(x)

        return x


class RNNBasedEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.rnn_units_0 = tf.keras.Sequential([
        #     RNNUnit(8, 8, return_sequence=True),
        #     RNNUnit(8, 1, return_sequence=True),
        # ])
        # self.rnn_units_1 = tf.keras.Sequential([
        #     RNNUnit(8, 8, ),
        #     RNNUnit(8, 1, ),
        # ])
        self.rnn_units_0 = RNNUnit(8, 8, return_sequence=True)
        self.rnn_units_1 = RNNUnit(8, 8, return_sequence=True)
        self.flatten = layers.Flatten()
        self.crossAttn = CrossAttention(d_model=20)
    def call(self,x, training=True):
        x_0, x_1=divide_x(x)
        shape = tf.shape(x_0)
        # x_0 = tf.reshape(x_0, (shape[0], shape[1]*shape[2], 1))
        # x_1 = tf.reshape(x_1, (shape[0], shape[1] * shape[2], 1))
        _, h_real  = self.rnn_units_0(x_0, training=training)
        _, h_imag  = self.rnn_units_1(x_1, training=training)
        # tf.print(tf.shape(h_imag))
        x = self.crossAttn(h_real, h_imag)
        # x = self.flatten(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, d_model=50, **kwargs):
        super().__init__(**kwargs)
        self.encImag = EncoderReal(d_model=d_model, num_heads=8)
        self.encReal = EncoderReal(d_model=d_model, num_heads=8)
        self.flatten = layers.Flatten()
        self.crossAttn = CrossAttention()
    def call(self, x):
        # x_real, x_imag = x[..., :64], x[..., 64:]
        x_real, x_imag = x[..., :66], x[..., 66:]
        # x_real, x_imag=create_x(x_real, x_imag)
        h_real = self.encReal(x_real)
        h_imag = self.encImag(x_imag)
        x = self.crossAttn(h_real, h_imag)
        x=self.flatten(x)
        return x

class ClassifierHead(layers.Layer):
    def __init__(self, d_model=16,**kwargs):
        super().__init__(**kwargs)
        self.cls = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ], name='classifier_head')

    def call(self, x):
        x=self.cls(x)

        return x

def get_augmenter(noise_std):

    return keras.Sequential(
    [
        layers.GaussianNoise(noise_std)
    ])


def divide_x(x):
    x_0, x_1 = x[..., :66], x[..., 66:]
    return x_0, x_1


def create_x(x1, x2):
    # x1 = tf.gather(x1, indices=[8,15,16,23,24, 43, 52, 53,60,61], axis=3)
    # x2 = tf.gather(x2, indices=[8, 15, 16, 23, 24, 43, 52, 53, 60, 61], axis=3)
    return x1, x2

class CNNAttnTD(Model):
    def __init__(self, d_model=16):
        super(CNNAttnTD, self).__init__()


        self.cls = ClassifierHead()
        # self.encoder = Encoder(name='karen')
        self.encoder = RNNBasedEncoder()
        train_augmentation = {"noise_std": 0.25 }
        # classification_augmentation = {"min_area": 0.75 }
        self.temperature = 0.1
        # self.train_augmenter = get_augmenter(**train_augmentation)
        # self.classification_augmenter = get_augmenter(**classification_augmentation)

        self.normalizer_real = tf.keras.layers.Normalization(axis=2)
        self.normalizer_imag = tf.keras.layers.Normalization(axis=2)



    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name="acc")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # x_real, x_imag, y = data['data_f_real'], data['data_f_real'], data['label_attsrc']
        # x_real, x_imag, y = data['eeg_f_real'], data['eeg_f_imag'], data['att_gender']
        # x_real, x_imag, y = data['data'], data['data'], data['label_lr']
        # x_real, x_imag, y = data['eeg'], data['eeg'], data['att_gender']
        # x_real = self.normalizer_real(x_real)
        # x_imag = self.normalizer_imag(x_imag)
        # x = tf.concat([x_real[..., tf.newaxis], x_imag[..., tf.newaxis]], axis=-1)
        # x = tf.concat([x_real, x_imag], axis=-1)

        # x = self.train_augmenter(x)
        y = data['label_lr']
        data['data'] = layers.GaussianNoise(2)(data['data'])
        # tf.print(data['eeg'])
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            y_pred = self(data)
            # x = self.encoder(data)
            # y_pred = self.cls(x)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights +self.cls.trainable_weights,
        )
        # gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights +self.cls.trainable_weights,
            )
        )
        # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.accuracy_metric.update_state(y, y_pred)
        self.loss_metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # x_real, x_imag, y = data['data_f_real'], data['data_f_real'], data['label_attsrc']
        # x_real, x_imag, y = data['eeg_f_real'], data['eeg_f_imag'], data['att_gender']
        # x_real, x_imag, y = data['data'], data['data'], data['label_lr']
        # x_real, x_imag, y = data['eeg'], data['eeg'], data['att_gender']
        # x_real = self.normalizer_real(x_real)
        # x_imag = self.normalizer_imag(x_imag)
        # x = tf.concat([x_real, x_imag], axis=-1)
        y=data['label_lr']

        y_pred = self(data)
        loss = self.compiled_loss(y, y_pred)
        self.accuracy_metric.update_state(y, y_pred)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


    def call(self, data):
        x_real, x_imag = data['data'], data['data']
        x_real = self.normalizer_real(x_real)
        x_imag = self.normalizer_imag(x_imag)
        x = tf.concat([x_real, x_imag], axis=-1)
        x = self.encoder(x)
        x = self.cls(x)
        # y_pred = self(x)

        return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_metric,
            self.accuracy_metric,
        ]

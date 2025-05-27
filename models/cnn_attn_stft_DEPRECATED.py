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
        self.cnn_in = layers.Conv2D(in_ch, kernels, strides = (1,1), padding='same', activation='relu')
        self.cnn_in_dropout = layers.Dropout(dropout)
        self.cnn_mid = layers.Conv2D((in_ch+out_ch)//2, kernels, strides=(1, 1), padding='same', activation='relu')
        self.cnn_mid_dropout = layers.Dropout(dropout)
        self.cnn_out = layers.Conv2D(out_ch, kernels, strides=(1, 1), padding='same', activation='relu')

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

class SelfAttention(layers.Layer):
    def __init__(self, num_heads=1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        h = self.mha(query = x, key= x, value= x)
        h = self.add([h,x])
        x = self.layernorm(h)

        return x

class CrossAttention(layers.Layer):
    def __init__(self, num_heads=1):
        super().__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, contex):
        h = self.mha(query = x, key= contex, value= contex)
        h = self.add([h,x])
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
            SelfAttention(d_model),
            SelfAttention(d_model),
            SelfAttention(d_model),
            SelfAttention(d_model),
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
            ConvSkip(1),
        ])
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
        ])

    def call(self, x):
        x = self.convs(x)
        known_axes = [i for i, size in enumerate(x.shape) if size == 1]

        x = tf.squeeze(x, axis=known_axes)
        x = self.pos_emb(x)
        # x += h
        x = self.attns(x)

        return x

class Encoder(layers.Layer):
    def __init__(self, d_model=50):
        super().__init__()
        self.encImag = EncoderReal(d_model=d_model, num_heads=8)
        self.encReal = EncoderReal(d_model=d_model, num_heads=8)
        self.flatten = layers.Flatten()
        self.crossAttn = CrossAttention()
    def call(self, x):
        x_real, x_imag = x[..., :64], x[..., 64:]
        # x_real, x_imag = x[..., :66], x[..., 66:]
        # x_real, x_imag=create_x(x_real, x_imag)
        h_real = self.encReal(x_real)
        h_imag = self.encImag(x_imag)
        x = self.crossAttn(h_real, h_imag)
        x=self.flatten(x)
        return x

class ClassifierHead(layers.Layer):
    def __init__(self, d_model=16):
        super().__init__()
        self.cls = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ])

    def call(self, x):
        x=self.cls(x)

        return x

def get_augmenter(noise_std):

    return keras.Sequential(
    [
        layers.GaussianNoise(noise_std)
    ])

def create_x(x1, x2):
    # x1 = tf.gather(x1, indices=[8,15,16,23,24, 43, 52, 53,60,61], axis=3)
    # x2 = tf.gather(x2, indices=[8, 15, 16, 23, 24, 43, 52, 53, 60, 61], axis=3)

    return x1, x2

class CNNAttnSTFT(Model):
    def __init__(self, d_model=16):
        super(CNNAttnSTFT, self).__init__()


        self.cls = ClassifierHead()
        self.encoder = Encoder()

        train_augmentation = {"noise_std": 0.25 }
        # classification_augmentation = {"min_area": 0.75 }
        self.temperature = 0.1
        self.train_augmenter = get_augmenter(**train_augmentation)
        # self.classification_augmenter = get_augmenter(**classification_augmentation)

        self.normalizer_real = tf.keras.layers.Normalization(axis=3)
        self.normalizer_imag = tf.keras.layers.Normalization(axis=3)




    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name="acc")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x_real, x_imag, y = data['data_f_real'], data['data_f_real'], data['label_attsrc']
        # x_real, x_imag, y = data['eeg_f_real'], data['eeg_f_imag'], data['att_gender']
        x_real = self.normalizer_real(x_real)
        x_imag = self.normalizer_imag(x_imag)
        # x = tf.concat([x_real[..., tf.newaxis], x_imag[..., tf.newaxis]], axis=-1)
        x = tf.concat([x_real, x_imag], axis=-1)

        # x = self.train_augmenter(x)
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # y_pred = self(x)
            x = self.encoder(x)
            y_pred = self.cls(x)
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
        x_real, x_imag, y = data['data_f_real'], data['data_f_real'], data['label_attsrc']
        # x_real, x_imag, y = data['eeg_f_real'], data['eeg_f_imag'], data['att_gender']
        x_real = self.normalizer_real(x_real)
        x_imag = self.normalizer_imag(x_imag)
        x = tf.concat([x_real, x_imag], axis=-1)
        y_pred = self(x)
        loss = self.compiled_loss(y, y_pred)
        self.accuracy_metric.update_state(y, y_pred)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


    def call(self, x):
        x = self.encoder(x)
        x = self.cls(x)

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

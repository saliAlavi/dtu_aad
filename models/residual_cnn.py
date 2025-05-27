import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, losses
from .layers import PaddedConv2D, apply_seq, td_dot, GEGLU
from models.base import BaseModel
from tensorflow.keras.models import Model


"""
From: https://github.com/divamgupta/stable-diffusion-tensorflow/
removed time embedding
"""
class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            # keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]

        self.out_layers = [
            # keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, x):
        x = apply_seq(x, self.in_layers)
        x = apply_seq(x, self.out_layers)
        # ret = self.skip_connection(x) + x
        return x


class Downsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.op = PaddedConv2D(channels, 3, stride=2, padding=0)

    def call(self, x):
        return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = PaddedConv2D(channels, 3, padding=0)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class ResCNNModel(Model):
    def __init__(self):
        super().__init__()
        self.input_blocks = [
            ResBlock(200, 200),
            ResBlock(150, 150),
            ResBlock(100, 100),
            ResBlock(80, 80),
            ResBlock(60, 60),
            ResBlock(30, 30),
        ]
        self.middle_block = [
            ResBlock(30, 30),
            Downsample(30),
            ResBlock(30, 30),
            Downsample(30),
        ]
        self.output_blocks = [
            ResBlock(30, 30),
            ResBlock(60, 60),
            ResBlock(80, 80),
            ResBlock(100, 100)
            ]
        self.downsample_layers=[
            ResBlock(50, 50),
            ResBlock(20, 20),
            Downsample(20),
            Downsample(20),
            ResBlock(10, 10),
            ResBlock(1, 1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid')
        ]

        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

        self.normalizer = layers.Normalization(axis=3)

    def call(self, x):
        def apply(x, layer):
            x = layer(x)
            return x

        # saved_inputs = []
        # for layer in self.input_blocks:
        #     x = apply(x, layer)
        #     saved_inputs.append(x)
        #
        # for layer in self.middle_block:
        #     x = apply(x, layer)
        #
        # for layer in self.output_blocks:
        #     x = apply(x, layer)

        for layer in self.downsample_layers:
            x = apply(x, layer)

        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['eeg_r'], data['label']
        x = self.normalizer(x)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.accuracy_metric.update_state(y, y_pred)

        tmp = {m.name: m.result() for m in self.metrics}
        tmp['loss'] = loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data['eeg_r'], data['label']
        x = self.normalizer(x)
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.accuracy_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    # def call(self, x):
    #     x = self.decoder(x)
    #     return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.accuracy_metric, ]
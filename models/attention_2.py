from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf


class SelfAttention(keras.layers.Layer):
    def __init__(self, d_h):
        super().__init__()
        self.attention = tf.keras.layers.Attention()
        self.dense = keras.layers.Dense(d_h, activation='relu')
        self.dropout = layers.Dropout(0.1)

    def call(self, x):
        h = self.attention([x, x])
        h = self.dense(h)
        # x = x+h
        x = self.dropout(x)
        return x


class Attention_2(Model):
    def __init__(self):
        super(Attention_2, self).__init__()
        self.normalizer = layers.Normalization(axis=1)
        self.normalizer_test = layers.Normalization(axis=2)
        self.latent_dim = 100
        self.encoder = tf.keras.Sequential([
            # layers.Flatten(),
            SelfAttention(300),
            SelfAttention(400),
            SelfAttention(100),
            SelfAttention(100),
            SelfAttention(100),
            SelfAttention(100),
            SelfAttention(100),
            SelfAttention(100),
            SelfAttention(20),
            SelfAttention(300),
            SelfAttention(100),
            SelfAttention(20),
            layers.Flatten(),
            layers.Dense(4, activation='sigmoid')
        ])

        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['eeg'], data['att_lr']
        x = self.normalizer(x)
        # noise = tf.random.normal([1,128,512], 0, 1, tf.float32)
        with tf.GradientTape() as tape:
            # x = x + noise * 2
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.accuracy_metric.update_state(y, y_pred)

        # tmp= {m.name: m.result() for m in self.metrics}
        # tmp['loss']=loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data['eeg'], data['att_lr']
        x = self.normalizer(x)
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.accuracy_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        x = self.encoder(x)

        return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.accuracy_metric,]
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
from tensorflow.keras.models import Model
"""
https://www.tensorflow.org/text/tutorials/transformer
"""
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


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Dense(d_model)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
"""
https://www.tensorflow.org/text/tutorials/transformer
"""
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

"""
https://www.tensorflow.org/text/tutorials/transformer
"""
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

"""
https://www.tensorflow.org/text/tutorials/transformer
"""
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(Model):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    self.out_layer = [  tf.keras.layers.Dense(256),
                        tf.keras.layers.Dense(128),
                        tf.keras.layers.Dense(64),
                        tf.keras.layers.Dense(32),
                        tf.keras.layers.Dense(4)]

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)

    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    x = tf.keras.layers.Flatten()(x)
    for i in range(5):
        x = self.out_layer[i](x)
    return x  # Shape `(batch_size, seq_len, d_model)`.

class AttentionEncoderModel(Encoder):
    def __init__(self):
        super(AttentionEncoderModel, self).__init__(num_layers=2, d_model=400, num_heads=4,
               dff=200, dropout_rate=0.1)

        self.normalizer = tf.keras.layers.Normalization(axis=1)
        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['eeg'], data['label']
        x = self.normalizer(x)
        # x=tf.expand_dims(x, axis=3)
        with tf.GradientTape() as tape:
            x = tf.transpose(x,perm=[0,2,1])
            x*=100
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
        x, y = data['eeg'], data['label']
        # x = tf.expand_dims(x, axis=3)
        x = self.normalizer(x)
        x = tf.transpose(x, perm=[0, 2, 1])
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
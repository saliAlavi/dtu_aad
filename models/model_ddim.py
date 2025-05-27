"""
From: https://github.com/divamgupta/stable-diffusion-tensorflow/
"""
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, losses
from .layers import PaddedConv2D, apply_seq, td_dot, GEGLU
from models.base import BaseModel
from tensorflow.keras.models import Model


class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5, groups=channels),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(out_channels),
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5, groups=out_channels),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x = inputs
        h = apply_seq(x, self.in_layers)
        # emb_out = apply_seq(emb, self.emb_layers)
        h = h #+ emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_head):
        super().__init__()
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head**-0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [keras.layers.Dense(n_heads * d_head)]

    def call(self, inputs):
        # assert type(inputs) is list
        # if len(inputs) == 1:
        #     inputs = inputs + [None]
        x=inputs
        context = x #if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = keras.layers.Permute((2, 1, 3))(q)  # (bs, num_heads, time, head_size)
        k = keras.layers.Permute((2, 3, 1))(k)  # (bs, num_heads, head_size, time)
        v = keras.layers.Permute((2, 1, 3))(v)  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attention = td_dot(weights, v)
        attention = keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, num_heads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        return apply_seq(h_, self.to_out)


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        x = inputs
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x)) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels, n_heads, d_head):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5, groups=channels)
        assert channels == n_heads * d_head
        self.proj_in = PaddedConv2D(n_heads * d_head, 1)
        self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head)]
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, inputs):
        x = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block(x)
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + x_in


class Downsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.op = PaddedConv2D(channels, 3, stride=2, padding=1)

    def call(self, x):
        return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class UNetModel(Model):
    def __init__(self):
        super().__init__()
        # self.time_embed = [
        #     keras.layers.Dense(1280),
        #     keras.activations.swish,
        #     keras.layers.Dense(1280),
        # ]
        self.input_blocks = [
            [PaddedConv2D(40, kernel_size=3, padding=1)],
            [ResBlock(40, 40), SpatialTransformer(40, 5, 8)],
            [ResBlock(40, 40), SpatialTransformer(40, 5, 8)],
            [Downsample(40)],
            [ResBlock(40, 80), SpatialTransformer(80, 5, 16)],
            [ResBlock(80, 80), SpatialTransformer(80, 5, 16)],
            [Downsample(80)],
            [ResBlock(80, 120), SpatialTransformer(120, 5, 24)],
            [ResBlock(120, 120), SpatialTransformer(120, 5, 24)],
            [Downsample(120)],
            [ResBlock(120, 120)],
            [ResBlock(120, 120)],
        ]
        self.middle_block = [
            ResBlock(120, 120),
            SpatialTransformer(120, 5, 24),
            ResBlock(120, 120),
        ]
        self.output_blocks = [
            [ResBlock(240, 120)],
            [ResBlock(240, 120)],
            [ResBlock(240, 120), Upsample(120)],
            [ResBlock(240, 120), SpatialTransformer(120, 5, 24)],
            [ResBlock(240, 120), SpatialTransformer(120, 5, 24)],
            [
                ResBlock(360, 240),
                SpatialTransformer(240, 5, 48),
                Upsample(240),
            ],
            [ResBlock(360, 80), SpatialTransformer(80, 5, 16)],  # 6
            [ResBlock(360, 80), SpatialTransformer(80,5, 16)],
            [
                ResBlock(120, 80),
                SpatialTransformer(80, 5, 16),
                Upsample(80),
            ],
            [ResBlock(120, 40), SpatialTransformer(40, 5, 8)],
            [ResBlock(120, 40), SpatialTransformer(40, 5, 8)],
            [ResBlock(120, 40), SpatialTransformer(40, 5, 8)],
        ]
        self.out = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]
        self.flatten= tf.keras.layers.Flatten()
        self.cls_out= tf.keras.layers.Dense(4)


        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

        self.normalizer = layers.Normalization(axis=2)

    def call(self, x):
        # x = tf.expand_dims( x, axis=3 )

        # emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            print(layer.name)
            print(layer)
            if isinstance(layer, ResBlock):
                x = layer(x)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x)
            else:
                x = layer(x)
            return x

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)

        for layer in self.middle_block:
            x = apply(x, layer)

        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                x = apply(x, layer)

        x = self.flatten(x)
        x = self.cls_out(x)
        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['eeg'], data['label']
        x = self.normalizer(x)
        x=tf.expand_dims(x, axis=3)
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
        x, y = data['eeg'], data['label']
        x = tf.expand_dims(x, axis=3)
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
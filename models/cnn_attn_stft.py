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
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class ConvUnetUnit(layers.Layer):
    def __init__(self, in_ch, out_ch, kernels = 3, dropout=0.1):
        super().__init__()
        self.cnn_in = layers.Conv2D(in_ch, kernels, strides = (1,1), padding='same', activation='swish')
        self.cnn_in_dropout = layers.Dropout(dropout)
        self.cnn_mid = layers.Conv2D((in_ch+out_ch)//2, kernels, strides=(1, 1), padding='same', activation='swish')
        self.cnn_mid_dropout = layers.Dropout(dropout)
        self.cnn_out = layers.Conv2D(out_ch, kernels, strides=(1, 1), padding='same', activation='swish')


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
            ConvSkip(1),
        ])
        self.attns = tf.keras.Sequential([
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

class EncoderGen(layers.Layer):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.pos_emb = PositionalEmbedding(d_model)
        self.convs = tf.keras.Sequential([
            ConvSkip(66),
            ConvSkip(55),
            # ConvSkip(44),
            # ConvSkip(33),
            # ConvSkip(22),
            # ConvSkip(11),
            # ConvSkip(22),
            ConvSkip(33),
            ConvSkip(20),
            ConvSkip(1),
        ])
        self.attns = tf.keras.Sequential([
            SelfAttention(num_heads=num_heads),
            layers.Dense(20, activation='relu'),
            layers.LayerNormalization(axis=-1),
            # SelfAttention(num_heads=num_heads),
            # layers.Dense(20, activation='relu'),
            # layers.LayerNormalization(axis=-1),
            # SelfAttention(num_heads=num_heads),
            # layers.Dense(20, activation='relu'),
            # layers.LayerNormalization(axis=-1),
            # SelfAttention(num_heads=num_heads),
            # layers.Dense(20, activation='relu'),
            # layers.LayerNormalization(axis=-1),
            # SelfAttention(num_heads=num_heads),
            # layers.Dense(20, activation='relu'),
            # layers.LayerNormalization(axis=-1),
        ])

    def call(self, x):
        x = self.convs(x)
        known_axes = [i for i, size in enumerate(x.shape) if size == 1]

        x = tf.squeeze(x, axis=known_axes)
        x = self.pos_emb(x)
        # x += h
        x = self.attns(x)
        # tf.print(tf.shape(x))
        return x


class Encoder(layers.Layer):
    def __init__(self, d_model=16):
        super().__init__()
        # self.encImag = EncoderImag(d_model)
        # self.encReal = EncoderReal(d_model)
        self.encImag =EncoderGen(d_model)
        self.encReal =EncoderGen(d_model)
        self.crossAttn = CrossAttention()
    def call(self, x):
        x_real, x_imag = x[..., :64], x[..., 64:]
        # x_real, x_imag = x[..., :5], x[..., 5:]
        h_real = self.encReal(x_real)
        h_imag = self.encImag(x_imag)
        x = self.crossAttn(h_real, h_imag)
        x=layers.Flatten()(x)
        return x

class ClassifierHead(layers.Layer):
    def __init__(self, d_model=16):
        super().__init__()
        self.cls = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ])

    def call(self, x):
        x=self.cls(x)

        return x

# class AttnCls(layers.Layer):
#     def __init__(self, d_model=16, num_heads=1):
#         super().__init__()
#         # self.encImag = EncoderImag(d_model)
#         # self.encReal = EncoderReal(d_model)
#         self.crossAttn = CrossAttention()
#         # self.cls = tf.keras.Sequential([
#         #     SelfAttention(),
#         #     layers.Flatten(),
#         #     layers.Dense(100, activation='swish'),
#         #     layers.Dense(50, activation='swish'),
#         #     layers.Dense(4, activation='sigmoid'),
#         # ])
#         self.cls = ClassifierHead()
#
#     def call(self, x):
#         x_real, x_imag = x[..., 0], x[..., 1]
#         # x_real, x_imag = x
#         h_real = self.encReal(x_real)
#         h_imag = self.encImag(x_imag)
#         x = self.crossAttn(h_real, h_imag)
#         x = self.cls(x)

        # return x

def get_augmenter(noise_std):
    data_width =  161
    data_height = 16
    data_channles = 66*2
    # zoom_factor = 1.0 - math.sqrt(min_area)
    # return keras.Sequential(
    #     [
    #         keras.Input(shape=(data_width, data_height, data_channles)),
    #         layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
    #         layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
    #     ]
    # )
    return keras.Sequential(
    [
        layers.GaussianNoise(noise_std)
    ])

class BatchCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='BatchCrossEntropy'):
        super(BatchCrossEntropy, self).__init__(reduction=reduction, name=name)

    def call(self, y_t,g_mat):
        n_classes = 2
        tf.debugging.assert_shapes([
            (y_t, ('N')),
            (g_mat, ('N', 'D')),
        ])
        batch_size = tf.shape(y_t)[0]
        y_t_oh = tf.one_hot(y_t, n_classes)
        y_t_gram = tf.linalg.matmul(y_t_oh, tf.transpose(y_t_oh, perm=[1,0]) )#-tf.eye(batch_size)
        tf.debugging.assert_shapes([
            (y_t_gram, ('N', 'N')),
        ])
        g_mat = g_mat-tf.math.reduce_max(g_mat, axis=1, keepdims=True)
        g_mat = tf.math.exp(g_mat)
        g_mat /= tf.math.reduce_sum(g_mat, axis=1, keepdims=True)

        # g_mat = tf.math.reduce_sum(g_mat, axis=1)
        epsilon=tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(g_mat, epsilon, 1.0 - epsilon)

        # Compute cross entropy loss
        loss = -tf.reduce_mean(y_t_gram * tf.math.log(y_pred) + (1 - y_t_gram) * tf.math.log(1 - y_pred))

        # g_mat = -tf.math.log(g_mat)#+tf.keras.backend.epsilon() )
        # g_mat = tf.multiply(g_mat, y_t_gram)
        # # tf.print(g_mat, summarize=-1)
        # loss = tf.reduce_mean(g_mat)#/2
        return loss

class SimilarityLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='SimilarityLoss'):
        super(SimilarityLoss, self).__init__(reduction=reduction, name=name)

    def call(self, y_t,g_mat):
        n_classes = 2
        tf.debugging.assert_shapes([
            (y_t, ('N')),
            (g_mat, ('N', 'D')),
        ])
        batch_size = tf.shape(y_t)[0]
        y_t_oh = tf.one_hot(y_t, n_classes)
        y_t_gram = tf.linalg.matmul(y_t_oh, tf.transpose(y_t_oh, perm=[1,0]) )#-tf.eye(batch_size)
        tf.debugging.assert_shapes([
            (y_t_gram, ('N', 'N')),
        ])
        g_mat = tf.math.exp(g_mat)
        g_mat /= tf.math.reduce_sum(g_mat, axis=1, keepdims=True)

        # g_mat = tf.math.reduce_sum(g_mat, axis=1)
        # tf.print('Start')
        # tf.print(g_mat, summarize=-1)
        # tf.print(y_t_gram, summarize=-1)
        epsilon=tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(g_mat, epsilon, 1.0 - epsilon)

        # Compute cross entropy loss
        loss = -tf.reduce_mean(y_t_gram * tf.math.log(y_pred) + (1 - y_t_gram) * tf.math.log(1 - y_pred))

        # g_mat = -tf.math.log(g_mat)#+tf.keras.backend.epsilon() )
        # g_mat = tf.multiply(g_mat, y_t_gram)

        # loss = tf.reduce_mean(g_mat)#/2
        return loss

def create_x(x1, x2):
    # x1 = tf.gather(x1, indices=[8,15,16,23,24, 43, 52, 53,60,61], axis=3)
    # x2 = tf.gather(x2, indices=[8, 15, 16, 23, 24, 43, 52, 53, 60, 61], axis=3)

    return x1, x2

class CNNAttnSTFTContrastive(Model):
    def __init__(self, d_model=16):
        super(CNNAttnSTFTContrastive, self).__init__()

        self.cls = ClassifierHead()
        self.projection_head = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(150),
            ],
            name="projection_head",
        )
        width = 161
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(10)], name="linear_probe"
        )
        self.encoder = Encoder()
        contrastive_augmentation = {"noise_std": 1 }
        classification_augmentation = {"noise_std": 0.25 }
        self.temperature = 0.1
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)

        self.normalizer_real = tf.keras.layers.Normalization(axis=2)
        self.normalizer_imag = tf.keras.layers.Normalization(axis=2)
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="acc", dtype=None)
        # self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

        #contrastive or classifier
        self.mode = 'classifier'

    @staticmethod
    def norm_zero(tensor):
        mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
        std  = tf.math.reduce_std(tensor, axis=1, keepdims=True)
        out = (tensor-mean)/std

        return out

    def contrastive_loss(self, projections_1, projections_2, y_t):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        similarities = self.norm_zero(similarities)

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here

        # loss = BatchCrossEntropy()(y_t, similarities)

        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        loss= (loss_1_2 + loss_2_1) / 2
        # loss_2_1 = BatchCrossEntropy()(y_t, tf.transpose(similarities))
        # loss_1_2 = keras.losses.sparse_categorical_crossentropy(
        #     contrastive_labels, similarities, from_logits=True
        # )
        # loss_2_1 = keras.losses.sparse_categorical_crossentropy(
        #     contrastive_labels, tf.transpose(similarities), from_logits=True
        # )
        # return (loss_1_2 + loss_2_1) / 2
        return loss



    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x_real, x_imag, y = data['data_f_real'], data['data_f_imag'], data['label_lr']
        # tf.print(tf.shape(x_real))
        # tf.print(tf.shape(x_imag))
        x_real, x_imag = create_x(x_real, x_imag)
        x_real = self.normalizer_real(x_real)
        x_imag = self.normalizer_imag(x_imag)
        # x = tf.concat([x_real[..., tf.newaxis], x_imag[..., tf.newaxis]], axis=-1)
        x = tf.concat([x_real, x_imag], axis=-1)
        augmented_data_1 = self.contrastive_augmenter(x)
        augmented_data_2 = self.contrastive_augmenter(x)
        with tf.GradientTape() as tape:
            #change the encoder to sequential
            features_1 = self.encoder(augmented_data_1, training=True)
            features_2 = self.encoder(augmented_data_2, training=True)
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            contrastive_loss = self.contrastive_loss(projections_1, projections_2, y)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        preprocessed_images = self.classification_augmenter(
            x, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            # class_logits = self.linear_probe(features, training=True)
            class_logits = self.cls(features, training=True)
            probe_loss = self.probe_loss(y, class_logits)

        gradients = tape.gradient(probe_loss, self.cls.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.cls.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(y, class_logits)

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        # loss = self.compiled_loss(y, y_pred)
        #
        # # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # self.accuracy_metric.update_state(y, y_pred)


        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_real, x_imag, y = data['data_f_real'], data['data_f_imag'], data['label_lr']
        x_real, x_imag = create_x(x_real, x_imag)
        x_real = self.normalizer_real(x_real)
        x_imag = self.normalizer_imag(x_imag)
        x = tf.concat([x_real, x_imag], axis=-1)
        # x = tf.concat([x_real[..., tf.newaxis], x_imag[..., tf.newaxis]], axis=-1)
        # y_pred = self(x, training=True)
        # self.compiled_loss(y, y_pred)
        # self.accuracy_metric.update_state(y, y_pred)
        augmented_data_1 = self.contrastive_augmenter(x)
        augmented_data_2 = self.contrastive_augmenter(x)
        features_1 = self.encoder(augmented_data_1, training=False)
        features_2 = self.encoder(augmented_data_2, training=False)

        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2, y)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        # class_logits = self.linear_probe(features, training=False)
        class_logits = self.cls(features_1, training=False)
        probe_loss = self.probe_loss(y, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(y, class_logits)
        return {m.name: m.result() for m in self.metrics}

    # def encoder(self, x):
    #     # x_real, x_imag = x[..., 0], x[..., 1]
    #     x_real, x_imag = x[..., :66], x[..., 66:]
    #     h_real = self.encReal(x_real)
    #     h_imag = self.encImag(x_imag)
    #     x = self.crossAttn(h_real, h_imag)
    #     x=layers.Flatten()(x)
    #     return x

    def call(self, x):
        # x_real, x_imag = x[..., 0], x[..., 1]
        x_real, x_imag = x[..., :64], x[..., 64:]

        h_real = self.encReal(x_real)
        h_imag = self.encImag(x_imag)
        x = self.crossAttn(h_real, h_imag)
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
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

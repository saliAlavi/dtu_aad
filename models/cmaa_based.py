import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow import keras

BATCH_SIZE = 16
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

  def call(self, x, training=False):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x /= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

# class ConvUnetUnit(layers.Layer):
#     def __init__(self, in_ch, out_ch, kernels = 3, dropout=0.1):
#         super().__init__()
#         self.cnn_in = layers.DepthwiseConv2D(kernels, strides = (1), padding='same', activation='relu')
#         self.cnn_in_dropout = layers.Dropout(dropout)
#         mid_layer_ch = (in_ch+out_ch)//4 if (in_ch+out_ch)//4>0 else 1
#         self.cnn_mid = layers.DepthwiseConv2D(kernels, strides=(1), padding='same', activation='relu')
#         self.cnn_mid_dropout = layers.Dropout(dropout)
#         self.cnn_out = layers.Conv2D(out_ch, kernels, strides=(1), padding='same', activation='relu')
#
#     def call(self, x, training=False):
#         x = self.cnn_in(x)+x
#         x = self.cnn_in_dropout(x, training)
#         x = self.cnn_mid(x)+x
#         x = self.cnn_mid_dropout(x, training)
#         x = self.cnn_out(x)
#
#         return x


class ConvUnetUnit(layers.Layer):
    def __init__(self, in_ch, out_ch, kernels = 3, dropout=0.1):
        super().__init__()
        self.cnn_in = layers.Conv1D(in_ch, kernels, strides = (1), padding='same', activation='relu')
        self.cnn_in_dropout = layers.Dropout(dropout)
        mid_layer_ch = (in_ch+out_ch)//4 if (in_ch+out_ch)//4>0 else 1
        self.cnn_mid = layers.Conv1D(mid_layer_ch, kernels, strides=(1), padding='same', activation='relu')
        self.cnn_mid_dropout = layers.Dropout(dropout)
        self.cnn_out = layers.Conv1D(out_ch, kernels, strides=(1), padding='same', activation='relu')

    def call(self, x, training=False):
        x = self.cnn_in(x)
        x = self.cnn_in_dropout(x, training)
        x = self.cnn_mid(x)
        x = self.cnn_mid_dropout(x, training)
        x = self.cnn_out(x)

        return x

class ConvSkip(layers.Layer):
    def __init__(self, ch, kernels = 3, dropout=0.1):
        super().__init__()
        self.cnn_in =  ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_mid = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_out = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)

    def call(self, x, training):
        # h =x
        x = self.cnn_in(x, training)
        x = self.cnn_mid(x, training) + x
        x = self.cnn_out(x, training) + x

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

    def call(self, x, trainign= False):
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

    def call(self, x, contex, training= False):
        q = self.query_emb(x)
        k = self.key_emb(contex)
        v = self.value_emb(contex)
        h = self.mha(query = q, key= k, value= v)
        h = self.add([h,q])
        x = self.layernorm(h)

        return x


# class EncoderImag(layers.Layer):
#     def __init__(self, d_model, num_heads=1):
#         super().__init__()
#         self.pos_emb = PositionalEmbedding(d_model)
#         self.convs = tf.keras.Sequential([
#             ConvSkip(50),
#             ConvSkip(25),
#             ConvSkip(1),
#         ])
#         self.attns = tf.keras.Sequential([
#             SelfAttention(d_model,num_heads=num_heads),
#             SelfAttention(d_model,num_heads=num_heads),
#             SelfAttention(d_model,num_heads=num_heads),
#             SelfAttention(d_model,num_heads=num_heads),
#         ])
#
#     def call(self, x, taining):
#         x = self.convs(x)
#         known_axes = [i for i, size in enumerate(x.shape) if size == 1]
#
#         x = tf.squeeze(x, axis=known_axes)
#         x = self.pos_emb(x)
#         x = self.attns(x)
#
#         return x

class AudioBlock(layers.Layer):
    def __init__(self, d_model, num_heads=1, window_size = 100):
        super().__init__()
        self.pos_emb = PositionalEmbedding(d_model)
        # self.convs = tf.keras.Sequential([
        #     ConvSkip(50),
        #     ConvSkip(25),
        #     ConvSkip(1),
        # ], name= 'conv_encreal')
        self.conv_0 = ConvSkip(50)
        self.conv_1 = ConvSkip(25)
        self.conv_2 = ConvSkip(1)

        self.attn_0 = SelfAttention(num_heads=num_heads)
        self.attn_1 = SelfAttention(num_heads=num_heads)
        self.attn_2 = SelfAttention(num_heads=num_heads)

        self.window_size = window_size
        # self.attns = tf.keras.Sequential([
        #     SelfAttention(num_heads=num_heads),
        #     SelfAttention(num_heads=num_heads),
        # ],name='attn_encreal')

    def call(self, x, training=False):
        x = tf.math.abs(x)
        weights = tf.ones(self.window_size, dtype=tf.float32) / self.window_size

        # Reshape the weights to match the input format of conv1d
        weights = tf.reshape(weights, [self.window_size, 1, 1])

        # Apply the moving average to the signal
        x = tf.nn.conv1d(x[:, :, tf.newaxis], weights, stride=1, padding='SAME')

        x = self.conv_0(x, training = training)
        x = self.conv_1(x, training = training)
        x = self.conv_2(x, training = training)
        known_axes = [i for i, size in enumerate(x.shape) if size == 1]
        x = tf.squeeze(x, axis=known_axes)
        x = self.pos_emb(x)
        x = self.attn_0(x, training=training)
        x = self.attn_1(x, training=training)
        x = self.attn_2(x, training=training)
        return x

class EncoderAudio(layers.Layer):
    def __init__(self, d_model=50, **kwargs):
        super().__init__(**kwargs)
        self.left_audio_enc = AudioBlock(d_model=d_model, window_size=100)
        self.right_audio_enc = AudioBlock(d_model=d_model, window_size=100)
        self.flatten = layers.Flatten()
        self.crossAttn = CrossAttention()
    def call(self, x, training = False):
        x_audio_l, x_audio_l = x['audio_l'], x['audio_r']
        h_left = self.encReal(x_audio_l, training = training)
        h_right = self.encImag(x_audio_l, training = training)
        x = self.crossAttn(h_left, h_right, training= training)
        x=self.flatten(x)
        return x

# class CNNEncoder(layers.Layer):
#     def __init__(self, d_model=50, **kwargs):
#         super().__init__(**kwargs)
#         self.encImag = EncoderReal(d_model=d_model, num_heads=8)
#         self.encReal = EncoderReal(d_model=d_model, num_heads=8)
#         self.flatten = layers.Flatten()
#         self.crossAttn = CrossAttention()
#     def call(self, x, training = False):
#         x_real, x_imag = x['data_f_real'], x['data_f_imag']
#         h_real = self.encReal(x_real, training = training)
#         h_imag = self.encImag(x_imag, training = training)
#         x = self.crossAttn(h_real, h_imag, training= training)
#         x=self.flatten(x)
#         return x


# class RNNBasedEncoder(layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.rnn_units_0 = RNNUnit(8, 8, return_sequence=True)
#         self.rnn_units_1 = RNNUnit(8, 8, return_sequence=True)
#         self.flatten = layers.Flatten()
#         self.crossAttn = CrossAttention(d_model=20)
#     def call(self,x, training=True):
#         x_0 = x['data']
#         x_1 =  x['data']
#         _, h_real = self.rnn_units_0(x_0, training=training)
#         _, h_imag = self.rnn_units_1(x_1, training=training)
#         x = self.crossAttn(h_real, h_imag)
#         x = self.flatten(x)
#         return x


class RNNBasedEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn_units_0 = RNNUnit(8, 8, return_sequence=True)
        # self.rnn_units_1 = RNNUnit(8, 8, return_sequence=True)KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
        self.flatten = layers.Flatten()
        # self.crossAttn = CrossAttention(d_model=20)
    def call(self,x, training=True):
        x = x['data']
        # x_1 =  x['data']
        _, h_real = self.rnn_units_0(x, training=training)
        # _, h_imag = self.rnn_units_1(x_1, training=training)
        # x = self.crossAttn(h_real, h_imag)
        x = self.flatten(h_real)
        return x


# class Encoder(layers.Layer):
#     def __init__(self, d_model=50, **kwargs):
#         super().__init__(**kwargs)
#         self.encImag = EncoderReal(d_model=d_model, num_heads=8)
#         self.encReal = EncoderReal(d_model=d_model, num_heads=8)
#
#         self.flatten = layers.Flatten()
#         self.crossAttn = CrossAttention()
#     def call(self, x):
#         # x_real, x_imag = x[..., :64], x[..., 64:]
#         x_real, x_imag = x[..., :66], x[..., 66:]
#         # x_real, x_imag=create_x(x_real, x_imag)
#         h_real = self.encReal(x_real)
#         h_imag = self.encImag(x_imag)
#         x = self.crossAttn(h_real, h_imag)
#         x=self.flatten(x)
#         return x


class TFEncoder(layers.Layer):
    def __init__(self, d_model=50, **kwargs):
        super().__init__(**kwargs)
        self.time_encoder = RNNBasedEncoder()
        self.audio_encoder = EncoderAudio(d_model=d_model, window_size=100)
        self.time_embedding = layers.Dense(40)
        self.f_embedding = layers.Dense(40)
        self.flatten = layers.Flatten()
        self.selfAttn = SelfAttention()
        self.crossAttn = CrossAttention()
    def call(self, x, training=False):
        h_time = self.time_encoder(x, training = training)
        h_a = self.audio_encoder(x, training=training)
        # tf.print(tf.shape(h_a))
        # tf.print(tf.shape(h_time))
        # h_time = tf.reshape(h_time, tf.shape(h_time))
        h_time = tf.expand_dims(h_time, -1)
        h_a = tf.expand_dims(h_a, -1)

        h_time = self.time_embedding(h_time)
        h_a = self.f_embedding(h_a)

        # x = tf.concat([h_time, h_a], axis=-2)
        # x = self.selfAttn(x)
        x = self.crossAttn(h_time, h_a)
        x = self.flatten(x)
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

    def call(self, x, training= False):
        x=self.cls(x)

        return x

class ProbeHead(layers.Layer):
    def __init__(self, d_model=16,**kwargs):
        super().__init__(**kwargs)
        self.cls = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ], name='classifier_head')

    def call(self, x, training= False):
        x=self.cls(x)

        return x


def get_augmenter(noise_std):

    return keras.Sequential(
    [
        layers.GaussianNoise(noise_std)
    ])


def divide_x(x):
    x_0, x_1 =x, x
    return x_0, x_1


def create_x(x1, x2):
    # x1 = tf.gather(x1, indices=[8,15,16,23,24, 43, 52, 53,60,61], axis=3)
    # x2 = tf.gather(x2, indices=[8, 15, 16, 23, 24, 43, 52, 53, 60, 61], axis=3)
    return x1, x2

class BatchCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='BatchCrossEntropy'):
        super(BatchCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.temperature=10

    # def call(self, y_t,g_mat):
    #     n_classes = 2
    #     tf.debugging.assert_shapes([
    #         (y_t, ('N')),
    #         (g_mat, ('N', 'D')),
    #     ])
    #     batch_size = tf.shape(y_t)[0]
    #     y_t_oh = tf.one_hot(y_t, n_classes)
    #     y_t_gram = tf.linalg.matmul(y_t_oh, tf.transpose(y_t_oh, perm=[1,0]) )#-tf.eye(batch_size)
    #     tf.debugging.assert_shapes([
    #         (y_t_gram, ('N', 'N')),
    #     ])
    #     g_mat = g_mat-tf.math.reduce_max(g_mat, axis=1, keepdims=True)
    #     g_mat = tf.math.exp(g_mat)
    #     g_mat /= tf.math.reduce_sum(g_mat, axis=1, keepdims=True)
    #
    #     # g_mat = tf.math.reduce_sum(g_mat, axis=1)
    #     epsilon=tf.keras.backend.epsilon()
    #     y_pred = tf.clip_by_value(g_mat, epsilon, 1.0 - epsilon)
    #
    #     # Compute cross entropy loss
    #     loss = -tf.reduce_mean(y_t_gram * tf.math.log(y_pred) + (1 - y_t_gram) * tf.math.log(1 - y_pred))
    #     return loss

    def call(self, projections_1, projections_2, y_t):
        #contrastive_loss
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

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
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        loss= (loss_1_2 + loss_2_1) / 2
        return loss

class Augmentation(layers.Layer):
    def __init__(self, noise=16,**kwargs):
        super().__init__(**kwargs)
        self.noise = layers.GaussianNoise(noise)

    def call(self, x, training= False):
        if training:
            x['data'] = self.noise(x['data'])
            x['data_f_real'] = self.noise(x['data_f_real'])
            x['data_f_imag'] = self.noise(x['data_f_imag'])
        return x



class AttnTFContrastiveAudio(Model):
    def __init__(self, d_model=16):
        super(AttnTFContrastiveAudio, self).__init__()

        self.encoder = TFEncoder()
        self.probe = ProbeHead()
        self.cls = ClassifierHead()
        # self.encoder = Encoder(name='karen')

        self.probe_augmentation = Augmentation(5)
        self.cls_augmentation = Augmentation(5)

        # self.enc_pipe = keras.Sequential([self.probe_augmentation, self.probe])

        train_augmentation = {"noise_std": 0.25 }
        # classification_augmentation = {"min_area": 0.75 }
        # self.temperature = 0.1
        # self.train_augmenter = get_augmenter(**train_augmentation)
        # self.classification_augmenter = get_augmenter(**classification_augmentation)

        self.normalizer_t = tf.keras.layers.Normalization(axis=2)
        self.normalizer_real = tf.keras.layers.Normalization(axis=3)
        self.normalizer_imag = tf.keras.layers.Normalization(axis=3)

        self.gnoise = layers.GaussianNoise(10)

        self.probe_loss = self.contrastive_loss
        self.classifier_loss = keras.losses.SparseCategoricalCrossentropy()

        self.lambda_=.9

        #Contrastive Parameters
        self.temperature = 0.1

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.loss_contrastive_metric = keras.metrics.Mean(name="p_loss")
        self.accuracy_contrastive_metric = keras.metrics.SparseCategoricalAccuracy(name="p_acc")
        self.loss_classifier_metric = keras.metrics.Mean(name="loss")
        self.accuracy_classifier_metric = keras.metrics.SparseCategoricalAccuracy(name="acc")

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
        y_t = data['label_lr']
        # data['data'] = layers.GaussianNoise(200)(data['data'])
        # data['data_f_real'] = layers.GaussianNoise(100)(data['data_f_real'], training=True)
        # data['data_f_real'] = layers.GaussianNoise(100)(data['data_f_real'], training=True)
        # tf.print(data['eeg'])
        data_0 = self.probe_augmentation(data)
        data_1 = self.probe_augmentation(data)
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # data['data'] = self.gnoise(data['data'])
            # data['data_f_real'] = self.gnoise(data['data_f_real'])
            # data['data_f_imag'] = self.gnoise(data['data_f_imag'])

            encoded_input_0 = self(data_0, training = True)
            encoded_input_1 = self(data_1, training=True)
            probe_head_0 = self.probe(encoded_input_0)
            probe_head_1 = self.probe(encoded_input_1)

            classifier_pred_0 = self.cls(encoded_input_0)
            classifier_pred_1 = self.cls(encoded_input_1)

            loss_probe = self.probe_loss(probe_head_0, probe_head_1, y_t)
            # x = self.encoder(data)
            # y_pred = self.cls(x)
            # loss = self.compiled_loss(y, y_pred)
            loss_cls_0 = self.classifier_loss(y_t, classifier_pred_0)
            loss_cls_1 = self.classifier_loss(y_t, classifier_pred_1)
            loss_cls = (loss_cls_0+loss_cls_1)/2

            loss = self.lambda_*loss_probe + (1-self.lambda_)*loss_cls

        # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights +self.cls.trainable_weights+self.probe.trainable_weights,
        )
        # gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights +self.cls.trainable_weights+self.probe.trainable_weights,
            )
        )
        # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.accuracy_classifier_metric.update_state(y_t, classifier_pred_0)
        self.accuracy_classifier_metric.update_state(y_t, classifier_pred_1)
        self.loss_classifier_metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # x_real, x_imag, y = data['data_f_real'], data['data_f_real'], data['label_attsrc']
        # x_real, x_imag, y = data['eeg_f_real'], data['eeg_f_imag'], data['att_gender']
        # x_real, x_imag, y = data['data'], data['data'], data['label_lr']
        # x_real, x_imag, y = data['eeg'], data['eeg'], data['att_gender']
        # x_real = self.normalizer_real(x_real)
        # x_imag = self.normalizer_imag(x_imag)
        # x = tf.concat([x_real, x_imag], axis=-1)
        y_t=data['label_lr']
        encoded_input = self(data, training = False)
        classifier_pred = self.cls(encoded_input)
        loss_cls = self.classifier_loss(y_t, classifier_pred)

        self.accuracy_classifier_metric.update_state(y_t, classifier_pred)
        self.loss_classifier_metric.update_state(loss_cls)
        return {m.name: m.result() for m in self.metrics}


    def call(self, data, training):
        # x_real, x_imag = data['data'], data['data']
        # data['data_f_real'] = self.normalizer_real(data['data_f_real'])
        # data['data_f_imag'] = self.normalizer_imag(data['data_f_imag'])
        # data['data']= self.normalizer_t(data['data'])
        # x = tf.concat([x_real, x_imag], axis=-1)
        if training:
            # data['data_f_real'] = self.gnoise(data['data_f_real'])
            # data['data_f_imag'] = self.gnoise(data['data_f_imag'])
            # data['data'] = self.gnoise(data['data'])
            x = self.encoder(data, training=training)
            # x = self.cls(x, training=training)
        else:
            x = self.encoder(data, training=training)
            # x = self.cls(x, training=training)

        return x

    def contrastive_loss(self, projections_1, projections_2, y_t):
        # contrastive_loss
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.accuracy_contrastive_metric.update_state(contrastive_labels, similarities)
        self.accuracy_contrastive_metric.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here

        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        loss = (loss_1_2 + loss_2_1) / 2

        self.loss_contrastive_metric(loss)

        return loss

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_contrastive_metric,
            self.accuracy_contrastive_metric,
            self.loss_classifier_metric,
            self.accuracy_classifier_metric,
        ]

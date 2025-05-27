import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow import keras
import copy

BATCH_SIZE = 64

PROBE_DIM = 50
PROBE_LATENT_DIM=PROBE_DIM//2

#wd from https://keras.io/examples/vision/simsiam/
WEIGHT_DECAY = 0.0005

D_MODEL = 128

def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)a
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbeddingBasic(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x, training=False):
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class ConvUnetUnit(layers.Layer):
    def __init__(self, in_ch, out_ch, kernels=3, dropout=0.1):
        super().__init__()
        self.cnn_in = layers.Conv1D(in_ch, kernels, strides=(1), padding='same', activation='relu')
        self.cnn_in_dropout = layers.Dropout(dropout)
        mid_layer_ch = (in_ch + out_ch) // 4 if (in_ch + out_ch) // 4 > 0 else 1
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
    def __init__(self, ch, kernels=3, dropout=0.1):
        super().__init__()
        self.cnn_in = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_mid = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)
        self.cnn_out = ConvUnetUnit(ch, ch, kernels=kernels, dropout=dropout)

    def call(self, x, training):
        # h =x
        x = self.cnn_in(x, training)
        x = self.cnn_mid(x, training) + x
        x = self.cnn_out(x, training) + x

        return x


class RNNUnit(layers.Layer):
    def __init__(self, rnn_units=32, in_dim=32, out_dim=32, dropout=0, return_sequence=True, **kwargs):
        super().__init__(**kwargs)
        self.return_sequence = return_sequence
        self.dense = layers.Dense(in_dim)
        # self.gru = tf.keras.layers.GRU(rnn_units,
        #                                return_sequences=True,
        #                                return_state=True)
        self.gru = layers.GRU(rnn_units, return_sequences=return_sequence, return_state=return_sequence,
                              dropout=dropout)
        self.dense = layers.Dense(out_dim)

    def call(self, x, training=True):
        # states = self.gru.get_initial_state(x)
        # x, states = self.gru(x, initial_state=states, training=training)
        sequence, x = self.gru(x, training=training)
        x = self.dense(x)
        if self.return_sequence:
            return x, sequence
        else:
            return x


class SelfAttention(layers.Layer):
    def __init__(self, d_model=50, num_heads=1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.query_emb = layers.Dense(d_model)
        self.key_emb = layers.Dense(d_model)
        self.value_emb = layers.Dense(d_model)

    def call(self, x, trainign=False):
        q = self.query_emb(x)
        k = self.key_emb(x)
        v = self.value_emb(x)
        h = self.mha(query=q, key=k, value=v)
        # h = self.add([h,q])
        # x = self.layernorm(h)
        x = h
        return x


class CrossAttention(layers.Layer):
    def __init__(self, d_model=50, num_heads=1):
        super().__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads)
        self.add = layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.query_emb = layers.Dense(d_model)
        self.key_emb = layers.Dense(d_model)
        self.value_emb = layers.Dense(d_model)

    def call(self, x, contex, training=False):
        q = self.query_emb(x)
        k = self.key_emb(contex)
        v = self.value_emb(contex)
        h = self.mha(query=q, key=k, value=v)
        h = self.add([h, q])
        x = self.layernorm(h)

        return x


class RNNBasedEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn_units_0 = RNNUnit(8, 8, return_sequence=True)
        # self.rnn_units_1 = RNNUnit(8, 8, return_sequence=True)
        self.flatten = layers.Flatten()
        # self.crossAttn = CrossAttention(d_model=20)

    def call(self, x, training=True):
        x = x['data']
        # x_1 =  x['data']
        _, h_real = self.rnn_units_0(x, training=training)
        # _, h_imag = self.rnn_units_1(x_1, training=training)
        # x = self.crossAttn(h_real, h_imag)
        x = self.flatten(h_real)
        return x


class EEGEncoder(layers.Layer):
    def __init__(self, d_model=16, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, training=False):
        x = x['eeg_csp']
        return x


class AudioEncoder(layers.Layer):
    def __init__(self, source='audio_m_ds', **kwargs):
        super().__init__(**kwargs)
        self.source = source

    def call(self, x, training=False):
        x = x[self.source]
        return x


class AECrossAttention(layers.Layer):
    def __init__(self, d_model=320, n_iters=5, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.cross_attn = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=2) for _ in range(n_iters)]
        self.ln_alpha = [layers.LayerNormalization() for _ in range(n_iters)]
        self.ln_beta = [layers.LayerNormalization() for _ in range(n_iters)]
        self.ln_ff = [layers.LayerNormalization() for _ in range(n_iters)]
        self.fc_0 = [layers.Dense(units=d_model, activation='relu') for _ in range(n_iters)]
        self.fc_1 = [layers.Dense(units=d_model, activation=None) for _ in range(n_iters)]
        self.pe = PositionalEmbeddingBasic(d_model=d_model)
        self.n = n_iters

    def call(self, audio_enc, eeg_enc, training=False):
        o_alpha = tf.expand_dims(audio_enc, axis=1)
        o_beta = eeg_enc
        for i in range(self.n):
            o_alpha_p = self.ln_alpha[i](o_alpha)
            o_beta_p = self.ln_beta[i](o_beta)
            o_beta_bar = self.cross_attn[i](query=o_beta_p, key=o_alpha_p, value=o_alpha_p)
            o_beta_bar = o_beta_bar + o_beta_p
            o_beta_bar = self.fc_1[i](self.fc_0[i](self.ln_ff[i](o_beta_bar))) + self.ln_ff[i](o_beta_bar)
            o_beta_bar = self.pe(o_beta_bar)
            o_beta = o_beta_bar

        x = o_beta
        return x


"""
Inputs:
    x: float [batch_size, x_channels, t_dim]
    y: float [batch_size, y_channels, t_dim]
Outputs:
    out: float [batch_size, x_channels, y_channels]
"""

class CosineSimilarity(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, y, training=False):
        x = tf.norm(x, ord=2)
        y = tf.norm(y, ord=2)
        out = tf.linalg.matmul(x, y)
        return out


class Classifier(layers.Layer):
    def __init__(self, d_model=16, **kwargs):
        super().__init__(**kwargs)
        self.cls = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(2, activation='sigmoid'),
        ], name='classifier_head')

    def call(self, x, training=False):
        x = self.cls(x)

        return x


class EncoderBase(layers.Layer):
    def __init__(self, d_aecross=320,**kwargs):
        super().__init__(**kwargs)
        self.eeg_encoder = EEGEncoder()
        self.audio_encoder_m = AudioEncoder(source='audio_m_ds')
        self.audio_encoder_f = AudioEncoder(source='audio_f_ds')
        self.cross_attention = AECrossAttention(d_model=d_aecross)
        self.cos_sim_m = CosineSimilarity()
        self.cos_sim_f = CosineSimilarity()
        self.flatten = layers.Flatten()
        self.dense_0 = layers.Dense(PROBE_DIM)

    def call(self, data, training=False):
        eeg = self.eeg_encoder(data)
        audio_m = self.audio_encoder_m(data)
        audio_f = self.audio_encoder_f(data)

        ca_m = self.cross_attention(audio_m, eeg)
        ca_f = self.cross_attention(audio_f, eeg)

        ca_m = layers.Flatten()(ca_m)
        ca_f = layers.Flatten()(ca_f)

        x = tf.concat([ca_m, ca_f], axis=-1)
        x = self.flatten(x)
        x = self.dense_0(x)

        return x


class ClassifierHead(layers.Layer):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)

        self.dense_0 = layers.Dense(100, activation='relu')
        self.do_0 = layers.Dropout(rate=dropout_rate)
        self.dense_1 = layers.Dense(50, activation='relu')
        self.do_1 = layers.Dropout(rate=dropout_rate)
        self.dense_2 = layers.Dense(2, activation='sigmoid', use_bias=False)

    def call(self, x, training=False):
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


class ProbeHead(layers.Layer):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.dense_0 = layers.Dense(PROBE_DIM, activation='relu', use_bias=False,  kernel_regularizer=regularizers.l2(WEIGHT_DECAY))
        self.do_0 = layers.Dropout(rate=dropout_rate)
        self.dense_1 = layers.Dense(PROBE_LATENT_DIM, activation='relu')
        self.do_1 = layers.Dropout(rate=dropout_rate)
        self.dense_2 = layers.Dense(PROBE_DIM, activation='relu', use_bias=True)

    def call(self, x, training=False):
        x = self.dense_0(x)
        # x = self.do_0(x, training=training)
        x = self.dense_1(x)
        # x = self.do_1(x, training=training)
        x = self.dense_2(x)
        return x



class AttnCrossAudioEEGSimsiam(Model):
    def __init__(self, d_aecross=320):
        super(AttnCrossAudioEEGSimsiam, self).__init__()

        self.train_contrastive = True

        self.encoder_base = EncoderBase(d_aecross)
        self.classifier_head = ClassifierHead(dropout_rate=0.05)
        self.probe = ProbeHead()

        self.normalizer_eeg = tf.keras.layers.Normalization(axis=1)
        self.normalizer_audio_m = tf.keras.layers.Normalization(axis=1)
        self.normalizer_audio_f = tf.keras.layers.Normalization(axis=1)

        self.classifier_loss = keras.losses.SparseCategoricalCrossentropy()

        self.lambda_ = .9

        # Contrastive Parameters
        self.temperature = 0.1

    def compile(self, opt_ctr=keras.optimizers.Adam(learning_rate=1e-3), opt_cls=keras.optimizers.Adam(learning_rate=1e-3), **kwargs):
        super().compile(**kwargs)

        self.opt_contrastive = opt_ctr
        self.opt_classifier = opt_cls

        self.loss_classifier_metric = keras.metrics.Mean(name="loss")
        self.accuracy_classifier_metric = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.loss_contrastive = keras.metrics.Mean(name="loss_ctr")
        self.accuracy_probe_metric = keras.metrics.SparseCategoricalAccuracy(name="acc_ctr")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        y_t = data['att_gender']
        # y_t = data['att_lr']
        # if self.train_contrastive:
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`

            # noise_std = 30000
            noise_std = 10
            data_0 = copy.copy(data)
            data_1 = copy.copy(data)
            data_0['eeg_csp'] =data_0['eeg_csp'] + tf.random.normal(tf.shape(data_0['eeg_csp']), mean=0,         stddev=noise_std)
            data_0['audio_m_ds']=data_0['audio_m_ds'] + tf.random.normal(tf.shape(data_0['audio_m_ds']), mean=0, stddev=noise_std)
            data_0['audio_f_ds']=data_0['audio_f_ds'] + tf.random.normal(tf.shape(data_0['audio_f_ds']), mean=0, stddev=noise_std)

            data_1['eeg_csp'] =  data_1['eeg_csp'] + tf.random.normal(   tf.shape(data_1['eeg_csp']), mean=0,    stddev=noise_std)
            data_1['audio_m_ds']=data_1['audio_m_ds'] + tf.random.normal(tf.shape(data_1['audio_m_ds']), mean=0, stddev=noise_std)
            data_1['audio_f_ds']=data_1['audio_f_ds'] + tf.random.normal(tf.shape(data_1['audio_f_ds']), mean=0, stddev=noise_std)

            p_0, z_0 = self(data_0, contrastive=True, training=True)
            p_1, z_1 = self(data_1, contrastive=True, training=True)

            loss_0 = self.SimSiameseLoss(p_0, z_1, y_t)
            loss_1 = self.SimSiameseLoss(p_1, z_0, y_t)
            loss = (loss_0+loss_1)/2


        # Compute gradients
        gradients = tape.gradient(
            loss,
            self.encoder_base.trainable_weights + self.probe.trainable_weights
        )

        self.opt_contrastive.apply_gradients(
            zip(
                gradients,
                self.encoder_base.trainable_weights + self.probe.trainable_weights
            )
        )

        self.loss_contrastive.update_state(loss)
        # classifier learning

        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`
            pred = self(data, contrastive = False, training=True)

            loss = self.classifier_loss(y_t, pred)

        # Compute gradients
        gradients = tape.gradient(
            loss,
            self.classifier_head.trainable_weights+self.encoder_base.trainable_weights
        )

        self.opt_classifier.apply_gradients(
            zip(
                gradients,
                self.classifier_head.trainable_weights+self.encoder_base.trainable_weights
            )
        )
        # Update weights
        # Update metrics (includes the metric that tracks the loss)
        # Return a dict mapping metric names to current value
        self.accuracy_classifier_metric.update_state(y_t, pred)
        self.loss_classifier_metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        y_t = data['att_gender']
        # y_t = data['att_lr']
        pred = self(data, contrastive=False, training=False)
        loss = self.classifier_loss(y_t, pred)

        self.accuracy_classifier_metric.update_state(y_t, pred)
        self.loss_classifier_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}
    
    def latent(self,data_inp):
        data = copy.copy(data_inp)
        data['eeg_csp'] = self.normalizer_eeg(data['eeg_csp'])
        data['audio_m_ds'] = self.normalizer_audio_m(data['audio_m_ds'])
        data['audio_f_ds'] = self.normalizer_audio_f(data['audio_f_ds'])

        p = self.encoder_base(data)

        return p

    def call(self, data_inp, training=False, contrastive=False):
        data = copy.copy(data_inp)
        data['eeg_csp'] = self.normalizer_eeg(data['eeg_csp'])
        data['audio_m_ds'] = self.normalizer_audio_m(data['audio_m_ds'])
        data['audio_f_ds'] = self.normalizer_audio_f(data['audio_f_ds'])
        

        if contrastive:
            if training:
                noise_std = 0.0
                data['eeg_csp'] = data['eeg_csp'] + tf.random.normal(tf.shape(data['eeg_csp']), mean=0,
                                                                     stddev=noise_std)
                data['audio_m_ds'] = data['audio_m_ds'] + tf.random.normal(tf.shape(data['audio_m_ds']), mean=0,
                                                                           stddev=noise_std)
                data['audio_f_ds'] = data['audio_f_ds'] + tf.random.normal(tf.shape(data['audio_f_ds']), mean=0,
                                                                           stddev=noise_std)
            p = self.encoder_base(data)
            z = self.probe(p, training=training)
            return p, z
        else:
            x = self.encoder_base(data)
            x = self.classifier_head(x, training=training)
            return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_classifier_metric,
            self.accuracy_classifier_metric,
            self.loss_contrastive,
            self.accuracy_probe_metric
        ]

    def SimCLRLoss(self, projections_1, projections_2, y_t):
        # contrastive_loss
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)
        # Cosine similarity: the dot product of the l2-normalized feature vectors

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        y_t_one_hot = tf.one_hot(y_t, 2)
        y_t_oh_gram = tf.matmul(y_t_one_hot, y_t_one_hot, transpose_b=True)

        similarities = tf.divide(tf.matmul(projections_1, projections_2, transpose_b=True), self.temperature)
        similarities = tf.divide(tf.math.exp(similarities), tf.reduce_sum(tf.math.exp(similarities), axis=1))
        loss = tf.math.multiply(y_t_oh_gram, similarities)
        loss = tf.reduce_sum(loss, axis=1)
        loss = -tf.math.log(loss)
        # tf.print(similarities)


        # tf.print(similarities)
        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        # batch_size = tf.shape(projections_1)[0]
        # contrastive_labels = tf.range(batch_size)
        # self.accuracy_probe_metric.update_state(contrastive_labels, similarities)
        # self.accuracy_probe_metric.update_state(
        #     contrastive_labels, tf.transpose(similarities)
        # )
        # # The temperature-scaled similarities are used as logits for cross-entropy
        # # a symmetrized version of the loss is used here
        # loss_1_2 = keras.losses.sparse_categorical_crossentropy(
        #     contrastive_labels, similarities, from_logits=True
        # )
        # loss_2_1 = keras.losses.sparse_categorical_crossentropy(
        #     contrastive_labels, tf.transpose(similarities), from_logits=True
        # )
        # loss= tf.reduce_mean((loss_1_2 + loss_2_1) / 2)
        # self.loss_contrastive.update_state(loss)

        return loss

    def SimCLRLoss_2(self, projections_1, projections_2, y_t):
        # contrastive_loss
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)
        # Cosine similarity: the dot product of the l2-normalized feature vectors

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = tf.divide(tf.matmul(projections_1, projections_2, transpose_b=True), self.temperature)

        # similarities = tf.divide(tf.math.exp(similarities), tf.reduce_sum(tf.math.exp(similarities), axis=1))
        # tf.print(similarities)
        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.accuracy_probe_metric.update_state(contrastive_labels, similarities)
        self.accuracy_probe_metric.update_state(
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
        loss = tf.reduce_mean((loss_1_2 + loss_2_1) / 2)
        self.loss_contrastive.update_state(loss)

        return loss

    #some excerpts from https://keras.io/examples/vision/simsiam/
    def SimSiameseLoss(self, p, z, y_t):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        y_t_one_hot = tf.one_hot(y_t, 2)
        y_t_one_hot_gram = tf.linalg.matmul(y_t_one_hot,y_t_one_hot,transpose_b=True)

        positive_pairwise = tf.divide(tf.reduce_sum(tf.linalg.matmul(p, z, transpose_b=True) *y_t_one_hot_gram, axis=1), tf.reduce_sum(y_t_one_hot_gram, axis=1))
        return -positive_pairwise

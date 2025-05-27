from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import tensorflow as tf

class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.normalizer = layers.Normalization(axis=2)
        self.latent_dim = 500
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(self.latent_dim, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(700, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(784, activation='relu'),
          layers.Dense(397, activation='relu'),
          layers.Dense(4, activation='sigmoid')
          # layers.Reshape((28, 28))
        ])

        self.accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=None)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data['eeg'], data['label']
        x = self.normalizer(x)
        with tf.GradientTape() as tape:
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

        tmp= {m.name: m.result() for m in self.metrics}
        tmp['loss']=loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data['eeg'], data['label']
        x = self.normalizer(x)
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.accuracy_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        x = self.decoder(x)
        return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.accuracy_metric,]
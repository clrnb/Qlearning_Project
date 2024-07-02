import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(64, 64))
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        q_values = self.fc2(x)
        return q_values

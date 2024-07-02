import random
from collections import deque

import numpy as np
import tensorflow as tf
from QNetwork import QNetwork

class Q_Agent:
    def __init__(self, env, max_steps, total_episodes, gamma=0.95, epsilon=1.0, epsilon_min=0.1,
                 replay_buffer_size=10000, batch_size=64, update_freq=8):
        self.max_steps = max_steps
        self.env = env
        self.total_episodes = total_episodes
        self.num_actions = 4
        self.gamma = gamma
        #Qnetwork
        self.q_network = QNetwork(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # Experience replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.steps = 0
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_network.call(np.expand_dims(state, axis=0)).numpy().flatten()
            return np.argmax(q_values)

    def update_Q(self, state, action, reward, next_state, done, episode):
        # Store transition in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.steps += 1

        # Perform experience replay update if buffer size is sufficient
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

            # Calculate the target Q-value based on the received reward and the maximum future reward
            # (if episode is done, the future reward is not considered)
            targets = rewards + (1 - dones) * self.gamma * np.max(self.q_network.call(next_states), axis=1)
            targets = targets.reshape(-1, 1)

            # Compute predicted Q-values for the chosen actions
            with tf.GradientTape() as tape:
                q_values = self.q_network.call(states)
                action_one_hot = tf.one_hot(actions, self.num_actions)
                predicted_q = tf.reduce_sum(q_values * action_one_hot, axis=1, keepdims=True)
                loss = self.loss_function(targets, predicted_q)

            # Backward pass (gradient calculation) and weight update
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))


            # Gradually decay epsilon
            decay_episodes = int(0.75 * self.total_episodes)
            if episode <= decay_episodes:
                epsilon_decay = (1.0 - self.epsilon_min) / decay_episodes
                epsilon = 1.0 - (episode - 1) * epsilon_decay
            else:
                epsilon = self.epsilon_min
            self.epsilon = max(epsilon, self.epsilon_min)

            #if done:
                #print("epsilon: ", self.epsilon)



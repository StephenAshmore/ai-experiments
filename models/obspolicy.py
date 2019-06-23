import tensorflow as tf
from tensorflow import keras

from .model import Model


class ObsPolicyModel(Model):
    # input_size should be a tuple like (128, 128, 3)
    # as in, 128 pixels by 128 pixels by 3 channels.
    def __init__(self, obs_block_count, state_size,
                 input_size, action_size, policy_block_count,
                 neurons):
        self.block_count = obs_block_count
        self.state_size = state_size
        self.input_size = input_size
        self.action_size = action_size
        self.policy_block_count = policy_block_count
        self.neurons = neurons or 64
        self.kernel_size = 6
        self.first_sample = False
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.previous_imagination_predictions = None
        self.previous_state_predictions = None
        self.previous_action_predictions = None

        if 2**obs_block_count >= self.neurons or 2**self.policy_block_count >= self.neurons:
            raise 'You must define enough layers to divide the starting neurons by two each layer.'
        

    def reset(self):
        self.first_sample = False
        self.previous_imagination_predictions = None
        self.previous_action_predictions = None

    def act(self, observations, reward, _done):
        observations = tf.cast(observations.reshape([1, 210, 160, 3]), tf.float32)
        if self.first_sample:
            # TODO:
            # Experiment with batching up all of these gradients to some batch size
            # say, collect the gradients for all of the models for 10 samples.
            with tf.GradientTape() as tape:
                # Backpropagate through imagination network
                # TODO: Using only the reward as our main loss signal
                # I want to use both reward + the imagination network's recreation of the image
                # but I do not know how to create the shape that would do that
                # it probably involves flattening the array.
                imagination_loss = self.loss_function(
                    reward, self.previous_imagination_predictions)

            # Apply gradients:
            grads = tape.gradient(
                imagination_loss, self.imagination_model.trainable_weights)
            self.opt_imagination.apply_gradients(
                zip(grads, self.imagination_model.trainable_weights))

            # back propagate error to inputs:
            # normalize the weights according to Chollet: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
            grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            if reward > 0:
                grads += 1 - (reward / 1000)
            else:
                grads += 1
            adjusted_actions = self.previous_action_predictions + grads

            # back propagate to policy network
            with tf.GradientTape() as tape:
                policy_loss = self.loss_function(
                    adjusted_actions, self.previous_action_predictions)

            grads = tape.gradient(
                policy_loss, self.policy_model.trainable_weights)
            self.opt_policy.apply_gradients(
                zip(grads, self.policy_model.trainable_weights))

            # back propagate error to state:
            # same as above
            grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            adjusted_state = self.previous_state_predictions + grads

            # backpropagate to observation network:
            with tf.GradientTape() as tape:
                observation_loss = self.loss_function(
                    adjusted_state, self.previous_state_predictions)

            grads = tape.gradient(
                observation_loss, self.observation_model.trainable_weights)
            self.opt_observation.apply_gradients(
                zip(grads, self.observation_model.trainable_weights))

            # don't need to go to the image level.

        # predict for next steps.
        state_predictions = self.observation_model(observations)
        action_predictions = self.policy_model(state_predictions)

        # compute input for imagination network:
        imag_input = tf.concat([action_predictions, state_predictions], 1)
        imagination_predictions = self.imagination_model(imag_input)

        # save predictions
        self.previous_imagination_predictions = imagination_predictions
        self.previous_state_predictions = state_predictions
        self.previous_action_predictions = action_predictions

        # get max action prediction:
        max_index, max_value = max(enumerate(action_predictions), key=lambda p: p[1])

        return max_index

    def build(self):
        self.observation_model = self.build_observation()
        self.policy_model = self.build_policy()
        self.imagination_model = self.build_imagination()

        self.opt_observation = keras.optimizers.Adam(lr=1e-3)
        # self.observation_model.compile(self.opt_observation, 'mse')
        self.opt_policy = keras.optimizers.Adam(lr=1e-4)
        # self.policy_model.compile(self.opt_policy, 'mse')
        self.opt_imagination = keras.optimizers.Adam(lr=1e-4)
        # self.imagination_model = keras.compile(self.opt_imagination, 'mse')

    def build_imagination(self):
        # given this state and these actions
        inp = keras.Input(shape=self.action_size + self.state_size)

        neurons = self.neurons

        out = keras.layers.Dense(neurons, activation='tanh')(inp)
        for i in range(self.policy_block_count):
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='tanh')(out)
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='tanh')(out)
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='relu')(out)
        # what will the image/future reward look like?
        out = keras.layers.Dense(1, activation='tanh')(out)
        return tf.keras.Model(inputs=inp, outputs=out)

    def build_policy(self):
        inp = keras.Input(shape=(self.state_size))

        neurons = self.neurons

        out = keras.layers.Dense(neurons, activation='tanh')(inp)
        for i in range(self.policy_block_count):
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='tanh')(out)
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='tanh')(out)
            out = keras.layers.Dense(
                neurons - (i + 1) * 12, activation='relu')(out)
        out = keras.layers.Dense(self.action_size, activation='tanh')(out)
        return tf.keras.Model(inputs=inp, outputs=out)

    def build_observation(self):
        with tf.device('cpu:0'):
            # filters = 12
            neurons = self.neurons

            inp = keras.Input(shape=self.input_size)
            out = keras.layers.Conv2D(neurons, (self.kernel_size, self.kernel_size))(inp)
        
            for _i in range(self.block_count):
                if neurons != self.neurons:
                    out = keras.layers.Conv2D(neurons, (self.kernel_size, self.kernel_size))(out)
                out = keras.layers.Dense(neurons, activation='relu')(out)
                out = keras.layers.MaxPooling2D()(out)
                neurons = int(neurons / 2)

            out = keras.layers.Flatten()(out)
            out = keras.layers.Dense(self.state_size, activation='tanh')(out)
        return tf.keras.Model(inputs=inp, outputs=out)

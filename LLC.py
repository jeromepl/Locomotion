import gym
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


LEARNING_RATE = 7e-4
N_STEPS = 5 # Number of actions to perform before reflecting on them (updating weights)
GAMMA = 0.95 # discount for past values, multiplies past Q-values, giving more importance to most recent values

# TODO
# https://github.com/keras-team/keras/blob/master/examples/image_ocr.py#L475
# For loss functions with extra parameters
# Also, it is possible to model.fit() with a generator as argument, which generates the training data batch-by-batch
# https://keras.io/models/model/#fit_generator
# Also, this might be useful:
# https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
# This if we need to dig deeper:
# https://github.com/keras-team/keras/issues/4746#issuecomment-269137712

class ActorCritic:
    def __init__(self, env):
        self.env = env

        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(256, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        actor_output = Dense(self.env.action_space.shape[0], activation='linear', name='actor_output')(h3)
        critic_output = Dense(self.env.action_space.shape[0], activation='linear', name='critic_output')(h3)

        # rand_normal = K.random_normal_variable(shape=actor_output.shape, mean=0, scale=0.1)
        # self.action_rand = actor_output + rand_normal
        
        model = Model(inputs=state_input, outputs=[actor_output, critic_output])
        adam  = Adam(lr=LEARNING_RATE)
        model.compile(optimizer=adam, loss={'actor_output': self.actor_loss_fn, 'critic_output': self.critic_loss_fn})
        model.summary()

        self.model = model
    
    def train(self, replay_buffer):
        replay_buffer = np.array(replay_buffer)
        nb_samples = len(replay_buffer)
        states = np.vstack(replay_buffer[:,0])
        actions = replay_buffer[:,1] # TODO this it not used because Keras recomputes the output when doing model.fit
        rewards = replay_buffer[:,2]
        values = np.vstack(replay_buffer[:,3])
        dones = replay_buffer[:,4]

        # Compute the "true" Q-values backwards from the last replay sample
        true_values = np.zeros((nb_samples, 1))
        for i in range(1, nb_samples + 1):
            if dones[-i]:
                true_values[-i] = 0 # Done means no more reward can be obtained from that point
                continue

            if i == 1:
                true_values[-i] = values[-1] # Start from the last sample, taking the Critic's value as base truth
            else:
                true_values[-i] = rewards[-i + 1] + GAMMA * true_values[-i + 1]

        advantages = true_values - values
        self.model.fit(states, {'actor_output': advantages, 'critic_output': true_values})

    # Evalute a single state and return both the actor and the critic outputs
    def evaluate(self, current_state, training=False):
        output = self.model.predict(np.array([current_state]))
        action = output[0][0]
        value = output[1][0]

        if training: # Explore the action space during training by adding a small noise to the action
            action = action + np.random.normal(0, 0.2, action.size) # TODO should the 'scale' be changed? - Probably not in MountainCar as the action is in [-1, 1]

        # Clip the action:
        action = np.array([min(max(action[0], -1), 1)])
        print(action)

        return action, value
    
    def actor_loss_fn(self, y_true, y_pred):
        # TODO the biggest thing here is that Keras recomputes the action outputted by the Actor network.
        # However, we not only already computed this value, we also added Gaussian noise to it to favour exploration of the action space...
        # y_pred = self.action

        advantages = y_true # the advantage is passed through y_true even though that is not the expected action output

        # Normalize between 0 and 1 and take into account the limits of the action space
        # This normalization is necessary in order to avoid NaN outputs of the 'log' operation
        # TODO I don't think this is working as intended
        # high = self.env.action_space.high[0] # TODO if the action space is not 1D, cannot use '[0]'
        # low = self.env.action_space.low[0]
        # normalized = (K.clip(y_pred, low, high) - low) / (high - low)
        normalized = K.clip(y_pred, self.env.action_space.low[0], self.env.action_space.high[0])

        # TODO subtract beta*entropy to encourage exploration
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html has an 'entropy()' method
        # return -K.log(normalized + 1e-10) * advantages
        return -K.log(normalized) * advantages
    
    def critic_loss_fn(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

def main():
    # Access the TensorBoard page in a browser through `tensorboard --logdir=./logs/`
    # TODO if used in the 'fit' method of Keras, this will generate a new graph for each time 'fit' is called, which is not what we want
    # tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0)

    env = gym.make('MountainCarContinuous-v0')

    ac = ActorCritic(env)
    # The replay buffer contains state (s), action (a), reward (r), value from Critic (v) and done (d) for each step taken
    replay_buffer = []

    while True: # for episode in range(5):
        state = env.reset()
        for t in range(1000):
            env.render()

            action, value = ac.evaluate(state, training=True)

            if len(replay_buffer) > 0:
                replay_buffer[-1][3] = value # Complete the previous sample by adding the 'next_value', which is now the current one

            state, reward, done, _ = env.step(action)

            replay_buffer.append([state, action, reward, 0, done]) # The 0 is a placeholder while we wait for the next_value to be calculated next iteration

            if done or len(replay_buffer) == N_STEPS + 1:
                training_samples = replay_buffer
                if not done:
                    training_samples = replay_buffer[:-1]
                ac.train(training_samples) # Reflect on the past N_STEPS actions
                replay_buffer = [replay_buffer[-1]] # Keep the last one since it is still waiting for its 'next_value' to be set

                if done:
                    print("Episode finished in {} steps".format(t + 1))
                    break

if __name__ == "__main__":
	main()
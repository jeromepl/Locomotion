import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import itertools


LEARNING_RATE = 1e-4
N_STEPS = 2 # Number of actions to perform before reflecting on them (updating weights)
GAMMA = 0.99 # discount for past values, multiplies past Q-values, giving more importance to most recent values
BETA = 1e-4 # entropy multiplier

SAVE_VIDEOS = False

class ActorCritic:
    def __init__(self, env, sess, summary_writer):
        self.env = env
        self.sess = sess
        self.summary_writer = summary_writer
        self.summary_step = 0

        with tf.name_scope('ac'):
            self._generate_network()
            self._generate_ops()
            self._generate_summaries()

    def _generate_network(self):
        self.state = tf.placeholder(tf.float32, shape=[None, self.env.observation_space.shape[0]])
        self.dense1 = tf.layers.Dense(units=128, activation=tf.nn.elu)(self.state)
        self.dense2 = tf.layers.Dense(units=64, activation=tf.nn.elu)(self.dense1)

        self.action_mu = tf.layers.Dense(units=self.env.action_space.shape[0], activation=tf.nn.tanh)(self.dense2)
        self.action_mu = tf.squeeze(self.action_mu)
        # Scale action_mu depending on the environment (since tanh activation yields (-1, 1))
        output_width = self.env.action_space.high[0] - self.env.action_space.low[0]
        self.action_mu *= output_width / 2
        self.action_mu += tf.constant(self.env.action_space.low[0] + output_width / 2, dtype=tf.float32)
        self.action_sigma = tf.layers.Dense(units=self.env.action_space.shape[0], activation=tf.nn.softplus)(self.dense2)
        self.action_sigma = tf.squeeze(self.action_sigma)

        self.value = tf.layers.Dense(units=1)(self.dense2)
        self.value = tf.squeeze(self.value)

        self.norm_dist = tf.contrib.distributions.Normal(self.action_mu, self.action_sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

    def _generate_ops(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.placeholder(tf.float32, name="advantage_train")
        self.actor_loss = -tf.log(self.norm_dist.prob(self.action_train) + 1e-10) * self.advantage_train - BETA * self.norm_dist.entropy()
        
        self.target_value = tf.placeholder(tf.float32, name="target_value")
        self.critic_loss = tf.losses.mean_squared_error(self.value, self.target_value)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_actor = self.optimizer.minimize(self.actor_loss)
        self.train_critic = self.optimizer.minimize(self.critic_loss)
    
    def _generate_summaries(self):
        tf.summary.scalar('Actor Loss', tf.reduce_sum(self.actor_loss))
        tf.summary.scalar('Critic Loss', tf.reduce_sum(self.critic_loss))
        tf.summary.histogram('Action mu', self.action_mu)
        tf.summary.histogram('Action sigma', self.action_sigma)
        # TODO tf.placeholder total_reward
        self.summary_op = tf.summary.merge_all() # scope='ac')
    
    def train(self, replay_buffer):
        replay_buffer = np.array(replay_buffer)
        states = np.vstack(replay_buffer[:,0])
        actions = np.vstack(replay_buffer[:,1])
        rewards = replay_buffer[:,2]
        values = np.vstack(replay_buffer[:,3])
        dones = replay_buffer[:,4]

        # Compute the "true" Q-values backwards from the last replay sample
        true_values = np.zeros(values.shape)
        for i in range(1, len(replay_buffer) + 1):
            if dones[-i]:
                true_values[-i] = 0 # Done means no more reward can be obtained from that point
                continue

            if i == 1:
                true_values[-i] = values[-1] # Start from the last sample, taking the Critic's value as base truth
            else:
                true_values[-i] = rewards[-i + 1] + GAMMA * true_values[-i + 1]

        advantages = true_values - values
        feed_dict = {
            self.state: states,
            self.action_train: actions,
            self.advantage_train: advantages,
            self.target_value: true_values
        }
        summary, _, _ = self.sess.run([self.summary_op, self.train_actor, self.train_critic], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step=self.summary_step)
        self.summary_step += 1

    # Evalute a single state and return both the actor and the critic outputs
    def __call__(self, current_state, training=False):
        feed_dict = { self.state: current_state }
        if training:
            return self.sess.run([self.action, self.value], feed_dict=feed_dict)
        else:
            return self.sess.run(self.action_mu, feed_dict=feed_dict) # Return the action before gaussian noise is applied

def main():
    env = gym.make('MountainCarContinuous-v0') # ('MountainCarContinuous-v0')
    # Save replay videos
    if SAVE_VIDEOS: # Affects performance
        video_dir = os.path.abspath('./videos')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = gym.wrappers.Monitor(env, video_dir, force=True)

    # Tensorboard config. Run `tensorboard --logdir=./logs`
    log_dir = os.path.abspath('./logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = tf.summary.FileWriter(log_dir)

    sess = tf.Session()

    ac = ActorCritic(env, sess, summary_writer)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    # The replay buffer contains state (s), action (a), reward (r), value from Critic (v) and done (d) for each step taken
    replay_buffer = []

    for episode in itertools.count():
        state = env.reset()
        total_reward = 0
        for t in itertools.count():
            env.render()

            state = np.expand_dims(state, 0)
            action, value = ac(state, training=True)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.append([state, action, reward, value, done])
            state = next_state
            total_reward += reward

            if done or len(replay_buffer) == N_STEPS:
                ac.train(replay_buffer) # Reflect on the past N_STEPS actions
                replay_buffer = []

            if done:
                if state[0] > 0.45:
                    print("Successfull after {} steps!".format(t+1))
                break

        print("Episode {} finished. Total reward: {}".format(episode, total_reward))
        # TODO tf.Summary.Value(tag='Total episode reward', simple_value=total_reward)

if __name__ == "__main__":
	main()
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Adapted from https://github.com/stefanbo92/A3C-Continuous

# PARAMETERS
RENDER = False              # Render one of the worker's environment
LOG_DIR = './logs'
SAVE_VIDEOS = False         # Save video replays
VIDEO_DIR = './videos'
N_WORKERS = 4               # Number of workers
GLOBAL_NET_SCOPE = 'Global_Net'
N_STEPS = 10                # Number of actions to perform before reflecting on them (updating weights)
GAMMA = 0.90                # Discount factor
ENTROPY_BETA = 0.01         # Entropy multiplier
LR_ACTOR = 0.0001           # Learning rate for actor
LR_CRITIC = 0.001           # Learning rate for critic

GAME = 'Pendulum-v0'
env = gym.make(GAME)
N_S = env.observation_space.shape[0]                    # Number of states
N_A = env.action_space.shape[0]                         # Number of actions
A_BOUND = [env.action_space.low, env.action_space.high] # Action bounds

# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None, summary_writer=None):
        self.scope = scope
        self.sess = sess
        self.globalAC = globalAC
        self.summary_writer = summary_writer
        self.summary_step = 0

        self.actor_optimizer = tf.train.RMSPropOptimizer(LR_ACTOR)
        self.critic_optimizer = tf.train.RMSPropOptimizer(LR_CRITIC)

        with tf.variable_scope(self.scope):
            self.state = tf.placeholder(tf.float32, [None, N_S])

            self._generate_network()

            if self.scope != GLOBAL_NET_SCOPE: # local network, calculate losses
                self._generate_ops()
                self._generate_summaries()
                
    def _generate_network(self): # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            dense1 = tf.layers.dense(self.state, 200, tf.nn.relu6, kernel_initializer=w_init)
            self.action_mu = tf.layers.dense(dense1, N_A, tf.nn.tanh, kernel_initializer=w_init) # estimated action value
            self.action_sigma = tf.layers.dense(dense1, N_A, tf.nn.softplus, kernel_initializer=w_init) # estimated variance

        # Scale action_mu depending on the environment (since tanh activation yields (-1, 1))
        # output_width = A_BOUND[1] - A_BOUND[0]
        # self.action_mu *= output_width / 2
        # self.action_mu += A_BOUND[1] + output_width / 2
        self.action_mu *= A_BOUND[1]
        self.action_sigma += 1e-4 # Ensure a minimum exploration

        with tf.variable_scope('critic'):
            dense2 = tf.layers.dense(self.state, 100, tf.nn.relu6, kernel_initializer=w_init)
            self.value = tf.layers.dense(dense2, 1, kernel_initializer=w_init)  # estimated value for state
        self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
    
    def _generate_ops(self):
        self.action_train= tf.placeholder(tf.float32, [None, N_A])        # action
        self.target_value = tf.placeholder(tf.float32, [None, 1]) # target_value value

        advantage = tf.subtract(self.target_value, self.value)
        self.critic_loss = tf.reduce_mean(tf.square(advantage))

        normal_dist = tf.contrib.distributions.Normal(self.action_mu, self.action_sigma)

        log_prob = normal_dist.log_prob(self.action_train)
        exp_v = log_prob * advantage
        entropy = normal_dist.entropy()  # encourage exploration
        self.exp_v = ENTROPY_BETA * entropy + exp_v
        self.actor_loss = tf.reduce_mean(-self.exp_v)

        self.action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1]) # sample a action from distribution

        self.actor_grads = tf.gradients(self.actor_loss, self.actor_params) #calculate gradients for the network weights
        self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

        self.pull_actor_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params, self.globalAC.actor_params)]
        self.pull_critic_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params, self.globalAC.critic_params)]
        self.update_actor_op = self.actor_optimizer.apply_gradients(zip(self.actor_grads, self.globalAC.actor_params))
        self.update_critic_op = self.critic_optimizer.apply_gradients(zip(self.critic_grads, self.globalAC.critic_params))

    
    def _generate_summaries(self):
        tf.summary.scalar('Actor Loss', tf.reduce_sum(self.actor_loss))
        tf.summary.scalar('Critic Loss', tf.reduce_sum(self.critic_loss))
        tf.summary.histogram('Action mu', self.action_mu)
        tf.summary.histogram('Action sigma', self.action_sigma)
        # TODO tf.placeholder total_reward
        self.summary_op = tf.summary.merge_all(scope=self.scope)

    def update_global(self, feed_dict):  # run by a local
        summary, _, _ = self.sess.run([self.summary_op, self.update_actor_op, self.update_critic_op], feed_dict)  # local grads applies to global net
        self.summary_writer.add_summary(summary, global_step=self.summary_step)
        self.summary_step += 1

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, state):  # run by a local
        state = state[np.newaxis, :]
        return self.sess.run(self.action, {self.state: state})[0]

# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, i, globalAC, summary_writer, sess):
        self.env = gym.make(GAME)   # make environment for each workers
        self.i = i
        self.AC = ACNet('worker{}'.format(i), sess, globalAC, summary_writer) # create ACNet for each worker
        self.sess = sess

        if SAVE_VIDEOS and i == 0: # Save replay videos. Affects performance
            video_dir = os.path.abspath(VIDEO_DIR)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.env = gym.wrappers.Monitor(self.env, video_dir, force=True)

        # Variables used to step through the simulation
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.state = self.env.reset()
        self.total_reward = 0

        self.episode_count = 0
        self.step_count = 0
   
    def step(self):
        if RENDER and self.i == 0:
            self.env.render()
        action = self.AC.choose_action(self.state)         # estimate stochastic action based on policy 
        next_state, reward, done, _ = self.env.step(action) # make step in environment

        self.total_reward += reward
        # save actions, states and rewards in buffer
        self.state_buffer.append(self.state)          
        self.action_buffer.append(action)
        self.reward_buffer.append((reward + 8.1368022) / 8.1368022)    # normalize reward between -1 and 1. The min reward is -16.2736044 and the max is 0.

        if self.step_count % N_STEPS == 0 or done:   # update global and assign to local net
            if done:
                true_value = 0   # terminal
            else:
                true_value = self.sess.run(self.AC.value, {self.AC.state: next_state[np.newaxis, :]})[0, 0]
            true_values = []
            for reward in self.reward_buffer[::-1]:    # reverse buffer r
                true_value = reward + GAMMA * true_value
                true_values.append(true_value)
            true_values.reverse()

            state_buffer, action_buffer, true_values = np.vstack(self.state_buffer), np.vstack(self.action_buffer), np.vstack(true_values)
            feed_dict = {
                self.AC.state: state_buffer,
                self.AC.action_train: action_buffer,
                self.AC.target_value: true_values,
            }
            self.AC.update_global(feed_dict) # actual training step, update global ACNet
            self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []
            self.AC.pull_global() # get global parameters to local ACNet

        self.state = next_state
        self.step_count += 1
        if done:
            print("Episode {} finished. Total reward for worker {}: {}".format(self.episode_count, self.i, self.total_reward))

            self.step_count = 0
            self.total_reward = 0
            self.episode_count += 1
            self.state = self.env.reset()

if __name__ == "__main__":

    log_dir = os.path.abspath(LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    
    sess = tf.Session()

    # with tf.device("/cpu:0"):
    global_ac = ACNet(GLOBAL_NET_SCOPE,sess)  # we only need its params
    workers = []
    # Create workers
    print ("Create {} workers".format(N_WORKERS))
    for i in range(N_WORKERS):
        workers.append(Worker(i, global_ac, summary_writer, sess))

    # coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    summary_writer.add_graph(sess.graph)

    while True:
        for worker in workers:
            worker.step()

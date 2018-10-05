#! /usr/bin/env python3
"""
CODE FROM https://github.com/pat-coady/trpo

PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from ppo.policy import Policy
from ppo.pf_policy import PFPolicy
from ppo.value_function import NNValueFunction
import scipy.signal
from ppo.utils import Logger, Scaler
from datetime import datetime
import time
import os
import argparse
import signal


# TODO
USE_PFNN = False

# How frequently to save tensorflow models to log folder (in number of episodes)
# NOTE This currently must be a multiple of the batch_size, otherwise no save will occur
NET_SAVE_FREQ = 5000


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, task_reward_weight, imitation_reward_weight, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, phases, actions, rewards, unscaled_obs = [], [], [], [], []
    task_rs, imitation_rs = [], []
    position_cost, velocity_cost, com_position_cost, com_velocity_cost = [], [], [], []
    done = False
    step = 0  # TODO try RSI
    phase = 0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale phase feature
    offset[-1] = 0.0  # don't offset phase feature
    while not done:
        if animate:
            env.render()

        # HIGHLIGHT
        obs = obs.astype(np.float32).reshape((1, -1))

        if not USE_PFNN:
            # add phase feature to observation
            obs = np.append(obs, [[phase]], axis=1)

        phases.append([[phase]])
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)

        if USE_PFNN:
            action = policy.sample(obs, [[phase]]).reshape(
                (1, -1)).astype(np.float32)
        else:
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)

        actions.append(action)
        # obs, reward, done, _ = env.step(np.squeeze(np.zeros(action.shape))) # Use this to test initialization
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))

        # Scale the task reward
        # reward = reward**task_reward_weight # TODO: Not safe to do on negative numbers (could get imaginary numbers)
        task_rs.append(reward)

        # Add the imitation reward to the standard env reward
        imitation_r, phase = env.env.env.robot.get_imitation_reward_and_phase(
            step)  # phase is [0, 1)
        reward /= (-imitation_r[0])**imitation_reward_weight
        imitation_rs.append(-(-imitation_r[0])**imitation_reward_weight)

        position_cost.append(imitation_r[1][0])
        velocity_cost.append(imitation_r[1][1])
        com_position_cost.append(imitation_r[1][2])
        com_velocity_cost.append(imitation_r[1][3])

        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 0.0165  # increment time step feature
        # ^ Value found in DeepMimicBaseBulletEnv.py in create_single_player_sc

    return (np.concatenate(observes), np.concatenate(phases), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs),
            np.array(task_rs), np.array(imitation_rs),
            (np.array(position_cost),
             np.array(velocity_cost),
             np.array(com_position_cost),
             np.array(com_velocity_cost)
             ))


def run_policy(env, policy, scaler, logger, episodes, task_reward_weight, imitation_reward_weight):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, phases, actions, rewards, unscaled_obs, task_r, imitation_r, imitation_r_logs = run_episode(
            env, policy, scaler, task_reward_weight, imitation_reward_weight)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'phases': phases,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs,
                      'task_r': task_r,
                      'imitation_r': imitation_r,
                      'imitation_r_logs': imitation_r_logs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    # update running statistics for scaling observations
    scaler.update(unscaled)
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    phases = np.concatenate([t['phases'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    # Cost ratios in imitation reward
    task_r = np.concatenate(
        [t['task_r'] for t in trajectories])
    imitation_r = np.concatenate(
        [t['imitation_r'] for t in trajectories])
    position_cost = np.concatenate(
        [t['imitation_r_logs'][0] for t in trajectories])
    velocity_cost = np.concatenate(
        [t['imitation_r_logs'][1] for t in trajectories])
    com_position_cost = np.concatenate(
        [t['imitation_r_logs'][2] for t in trajectories])
    com_velocity_cost = np.concatenate(
        [t['imitation_r_logs'][3] for t in trajectories])

    return observes, phases, actions, advantages, disc_sum_rew, task_r, imitation_r, (
        position_cost, velocity_cost, com_position_cost, com_velocity_cost)


def log_batch_stats(observes, actions, advantages, disc_sum_rew, task_r, imitation_r, imitation_r_logs, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_mean_task_reward': np.mean(task_r),
                '_mean_imitation_reward': np.mean(imitation_r),
                '_mean_position_cost': np.mean(imitation_r_logs[0]),
                '_mean_velocity_cost': np.mean(imitation_r_logs[1]),
                '_mean_com_position_cost': np.mean(imitation_r_logs[2]),
                '_mean_com_velocity_cost': np.mean(imitation_r_logs[3])
                })


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar, task_reward_weight, imitation_reward_weight, logfolder, timemax, jobid):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)

    start_time = time.time() # In seconds

    if not USE_PFNN:
        # add 1 to obs dimension for time step feature (see run_episode())
        obs_dim += 1

    if jobid is not None:
        path = os.path.join(logfolder, jobid)
    else:
        now = datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
        path = os.path.join(logfolder, now)
    logger = Logger(path)
    aigym_path = os.path.join(path, 'videos')
    env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)

    if USE_PFNN:
        policy = PFPolicy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
    else:
        policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)

    # Save a reference to both models in the logger in order to save the models once trained
    logger.add_tf_models({'policy': policy, 'value': val_func})

    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, 5,
               task_reward_weight, imitation_reward_weight)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(
            env, policy, scaler, logger, batch_size, task_reward_weight, imitation_reward_weight)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        # calculated discounted sum of Rs
        add_disc_sum_rew(trajectories, gamma)
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, phases, actions, advantages, disc_sum_rew, task_r, imitation_r, imitation_r_logs = build_train_set(
            trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages,
                        disc_sum_rew, task_r, imitation_r, imitation_r_logs, logger, episode)

        # update policy
        if USE_PFNN:
            policy.update(observes, phases, actions, advantages, logger)
        else:
            policy.update(observes, actions, advantages, logger)

        val_func.fit(observes, disc_sum_rew, logger)  # update value function

        logger.write(display=True)  # write logger results to file and stdout
        if episode % NET_SAVE_FREQ == 0:
            # Save the tensorflow models every once in a while
            logger.save_models(episode)

        # Stop execution if user pressed Ctrl+C or if maximum time has elapsed
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
        # Convert timemax from hours to seconds and give some room for error (30 min)
        # just to be safe in the case this is running with a time limit on compute canada
        if time.time() - start_time > max(timemax - 0.5, 1) * 3600:
            break
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-e', '--env_name', type=str, help='OpenAI Gym environment name',
                        default='DeepMimicHumanoidEnv-v0')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=200000)
    parser.add_argument('-g', '--gamma', type=float,
                        help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             ' (integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('-t', '--task_reward_weight', type=float,
                        help='Weight of the task reward',
                        default=1.0)
    parser.add_argument('-i', '--imitation_reward_weight', type=float,
                        help='Weight of the imitation reward',
                        default=1.0)
    parser.add_argument('-L', '--logfolder', type=str,
                        help='Path to the logs folder', default='D:\\ml-research-logs')
    parser.add_argument('-T', '--timemax', type=int,
                        help='Maximum time (in hours) allowed for this to run, after which'
                             ' program execution stops and data is saved. Note that'
                             ' execution will stop within 30 min of the allowed time in order'
                             ' to make sure that the last epoch fully runs', default=168) # 7 days default
    parser.add_argument('-j', '--jobid', type=str, default=None,
                        help='Slurm job ID to use as the name of the log folder')

    parser.add_argument('-r', '--render', type=bool, nargs='?',
                        help='Show the humanoid training process in a new window',
                        const=True, default=False)

    args = parser.parse_args()

    import gym_env.DeepMimicHumanoidGymEnv
    gym.envs.registration.register('DeepMimicHumanoidEnv-v0',
                                   entry_point='gym_env.DeepMimicHumanoidGymEnv:DeepMimicHumanoidGymEnv',
                                   max_episode_steps=1000,
                                   kwargs={
                                       'render': args.render
                                   })
    del args.render  # This argument should not be passed to the main function

    main(**vars(args))

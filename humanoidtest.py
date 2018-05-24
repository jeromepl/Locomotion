import gym
# import tensorflow as tf
import numpy as np
from pybullet_envs.bullet.simpleHumanoidGymEnv import SimpleHumanoidGymEnv


def main():
    env = gym.make('HumanoidBulletEnv-v0')
    # env = gym.wrappers.Monitor(env, "videos-humanoid", force=True)
    env.render(mode='human')
    env.reset()
    for _ in range(500):
        env.render(mode='human')
        _, _, done, _ = env.step(np.zeros(env.action_space.shape[0]))

        if done:
            env.reset()

if __name__ == '__main__':
    main()
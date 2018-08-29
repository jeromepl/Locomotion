from gym_env.DeepMimicBaseBulletEnv import DeepMimicBaseBulletEnv
from gym_env.Humanoid import Humanoid

# from pybullet_envs.bullet.robot_locomotors import Humanoid
# from pybullet_envs.bullet.gym_locomotion_envs import WalkerBaseBulletEnv

# TAKEN FROM: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py


class DeepMimicHumanoidGymEnv(DeepMimicBaseBulletEnv):
    def __init__(self, robot=Humanoid(), render=False):
        self.robot = robot
        DeepMimicBaseBulletEnv.__init__(self, self.robot, render=render)
        self.electricity_cost = 4.25 * DeepMimicBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * DeepMimicBaseBulletEnv.stall_torque_cost

        # self.reset()
        # robot2 = Humanoid()
        # robot2.scene = robot.scene
        # robot2.reset(robot._p)
        # robot2.robot_body.reset_position([0, 1, 0.8])
        # # robot2.robot_body.reset_orientation([0, 0, 0])

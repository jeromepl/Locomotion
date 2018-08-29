from gym_env.WalkerBase import WalkerBase
# from pybullet_envs.robot_locomotors import WalkerBase
import numpy as np

from imitation_reward import get_imitation_reward_and_phase, get_phase

# TAKEN FROM:https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py


class Humanoid(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self):
        WalkerBase.__init__(self,  'humanoid_symmetric_no_ground.xml',
                            'torso', action_dim=17, obs_dim=44, power=0.41)
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

    def get_imitation_reward_and_phase(self, elapsed_time):
        # TODO pass self._p to this function
        return get_imitation_reward_and_phase(self.robot_body.bodies[self.robot_body.bodyIndex],
                                              self.jdict, elapsed_time=elapsed_time)

    def get_phase(self, elapsed_time):
        return get_phase(elapsed_time)

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x",
                             "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x",
                             "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1",
                             "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            position = [0, 0, 0]
            orientation = [0, 0, 0]
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            if self.random_lean and self.np_random.randint(2) == 0:
                cpose.set_xyz(0, 0, 1.4)
                if self.np_random.randint(2) == 0:
                    pitch = np.pi/2
                    position = [0, 0, 0.45]
                else:
                    pitch = np.pi*3/2
                    position = [0, 0, 0.25]
                roll = 0
                orientation = [roll, pitch, yaw]
            else:
                position = [0, 0, 1.4]
                # just face random direction, but stay straight otherwise
                orientation = [0, 0, yaw]
            self.robot_body.reset_position(position)
            self.robot_body.reset_orientation(orientation)
        self.initial_z = 0.8

    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(
                float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying
        return +2 if z > 0.78 else -1

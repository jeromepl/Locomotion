# from gym_env.MJCFBaseBulletEnv import MJCFBaseBulletEnv
# from gym_env.StadiumScene import SinglePlayerStadiumScene

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
import numpy as np
import pybullet

# TAKEN FROM: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py


class DeepMimicBaseBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, robot, render=False):
        print("WalkerBase::__init__ start")
        MJCFBaseBulletEnv.__init__(self, robot, render)

        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(
            bullet_client, gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return self.stadium_scene

    def _reset(self):
        if (self.stateId >= 0):
            #print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv._reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                                                                                             self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            #print("saving state self.stateId:",self.stateId)

        return r

    def _isDone(self):
        return self._alive < 0

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        pose.move_xyz(init_x, init_y, init_z)
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    # cost for running electric current through a motor even at zero rotational speed, small
    stall_torque_cost = -0.1
    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_collision_cost = -1.0
    # to distinguish ground and other objects
    foot_ground_object_names = set(["floor"])
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # state[0] is body height above ground, body_rpy[1] is pitch
        self._alive = float(self.robot.alive_bonus(
            state[0]+self.robot.initial_z, self.robot.body_rpy[1]))
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        # TODO: Maybe calculating feet contacts could be done within the robot code
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost = self.electricity_cost * \
            float(np.abs(a*self.robot.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if(debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [  # HIGHLIGHT
            self._alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
            # self.robot.get_imitation_reward() * 10e16
        ]

        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

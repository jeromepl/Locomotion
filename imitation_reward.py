from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer

import pybullet as p
import numpy as np

# NOTE: This bvh was manually edited to only contain the portion of the recording
# from the moment the right foot touches the ground to the moment it touches it again
# after a full cycle of the movement. In DeepMimic, half of this motion is used and
# is mirrored for the left foot in order to have a full motion cycle
CMU_BVH_FILE = 'res/walk-02-01-cmu-cut.bvh'
BVH_RATE = 120  # in frames per second

# NOTE: This xml file was manually edited to add (fixed) joints for both hands,
# both feet and the head. This was necessary in order to obtain their positions
MUJOCO_XML_FILE = 'res/humanoid_symmetric_no_ground.xml'

# Mutiply mujoco positions values by this number in order to better match the CMU model
MUJOCO_SCALE_FACTOR = 16.9

# The scales are used to try to normalize each component of this reward (before applying the weights)
POSITION_COST_SCALE = 1.5
VELOCITY_COST_SCALE = 1/3
COM_POSITION_COST_SCALE = 22
COM_VELOCITY_COST_SCALE = 1.8

# Importance given to the position and velocity costs in the reward function
# For better readability, these values should add up to 1
POSITION_COST_WEIGHT = 0.65
VELOCITY_COST_WEIGHT = 0.15
COM_POSITION_COST_WEIGHT = 0.1  # Z-axis only pos difference of the center of masses
COM_VELOCITY_COST_WEIGHT = 0.1

# Scale the reward in order to better the other rewards defined in the environment
# NOTE: For the exact same DeepMimic implementation (with np.exp), a value of 1.43e16 seems to make sense
OVERALL_REWARD_SCALE = 0.022

DRAW_NAMES = True  # Whether to display extremity joint names when running the test

extremities = {
    'right_hand': {
        'cmu': {
            'joint_name': 'RightHand'
        },
        'mujoco': {
            'link_name': 'right_hand'
        }
    },
    'left_hand': {
        'cmu': {
            'joint_name': 'LeftHand'
        },
        'mujoco': {
            'link_name': 'left_hand'
        }
    },
    'right_foot': {
        'cmu': {
            'joint_name': 'RightFoot'
        },
        'mujoco': {
            'link_name': 'right_foot'
        }
    },
    'left_foot': {
        'cmu': {
            'joint_name': 'LeftFoot'
        },
        'mujoco': {
            'link_name': 'left_foot'
        }
    },
    'head': {
        'cmu': {
            'joint_name': 'Head_Nub'
        },
        'mujoco': {
            'link_name': 'head'
        }
    }
}

# NOTE: For mujoco, we could use the base position here, but that is not where the actual center of mass is
# example: root_y, root_x, root_z = p.getBasePositionAndOrientation(body)[0]
center_of_mass = {
    'cmu': {
        'joint_name': 'Hips'
    },
    'mujoco': {
        'joint_name': 'abdomen_x'
    }
}


def get_bvh_positions(bvh_file: str):
    parser = BVHParser()
    parsed_data = parser.parse(bvh_file)
    mp = MocapParameterizer('position')
    positions = mp.fit_transform([parsed_data])
    # Returns a pandas dataframe containing the positions of all frames
    return positions[0]


cmu_df = get_bvh_positions(CMU_BVH_FILE)  # Load the CMU mocap recording
frame_count = len(cmu_df.values)


def get_imitation_reward_and_phase(mujoco_body, mujoco_joints, elapsed_time=0, use_com_cost=True):
    return get_imitation_reward(mujoco_body, mujoco_joints, elapsed_time=elapsed_time, use_com_cost=use_com_cost), get_phase(elapsed_time)


def get_phase(elapsed_time: float) -> float:
    '''
    'elapsed_time' argument should be in seconds
    Returns a value in [0, 1) representing how far along the motion we currently are
    '''
    return ((elapsed_time * BVH_RATE) % frame_count) / frame_count


def get_imitation_reward(mujoco_body, mujoco_joints, elapsed_time=0, use_com_cost=True):
    '''
    Calculate a reward for imitating the reference CMU mocap motion. This reward is based on the following:
    - Distance between the end-effectors (hands, feet and head), relative to the center of mass, of
      the mujoco 3D model current state and the reference motion at the given time
    - Difference in the velocity of end-effectors
    - Difference in the velocity of the center of mass of both models
    - Difference in the height (Z-axis) of the center of mass of both models
    '''

    # elapsed_time / (1/BVH_RATE)
    frame = int(elapsed_time * BVH_RATE) % frame_count
    position_cost = 0
    velocity_cost = 0

    cmu_center_of_mass_joint = center_of_mass['cmu']['joint_name']
    cmu_root_pos = get_cmu_position(cmu_df, cmu_center_of_mass_joint, frame)
    cmu_root_vel = get_cmu_velocity(cmu_df, cmu_center_of_mass_joint, frame)

    mujoco_center_of_mass_joint_id = center_of_mass['mujoco'].get('joint_id')
    if mujoco_center_of_mass_joint_id is None:
        mujoco_center_of_mass_joint_id = get_and_save_mujoco_joint_id(
            mujoco_joints, center_of_mass['mujoco'])
    mujoco_root_pos = get_mujoco_position(
        mujoco_body, mujoco_center_of_mass_joint_id)
    mujoco_root_vel = get_mujoco_velocity(
        mujoco_body, mujoco_center_of_mass_joint_id)

    for extremity_key in extremities:
        extremity = extremities[extremity_key]

        cmu_joint = extremity['cmu']['joint_name']
        cmu_pos = get_cmu_position(cmu_df, cmu_joint, frame)
        cmu_vel = get_cmu_velocity(cmu_df, cmu_joint, frame)

        mujoco_joint_id = extremity['mujoco'].get('joint_id')
        if mujoco_joint_id is None:
            mujoco_joint_id = get_and_save_mujoco_joint_id(
                mujoco_joints, extremity['mujoco'])
        mujoco_pos = get_mujoco_position(mujoco_body, mujoco_joint_id)
        mujoco_vel = get_mujoco_velocity(mujoco_body, mujoco_joint_id)
        # NOTE we could also get the orientation and angular velocity through the link state

        # Compute the position cost
        cmu_dist_vector = cmu_pos - cmu_root_pos
        mujoco_dist_vector = mujoco_pos - mujoco_root_pos
        dist = np.linalg.norm(cmu_dist_vector - mujoco_dist_vector)
        position_cost += dist

        # Compute the velocity cost
        cmu_v_dist_vector = cmu_vel - cmu_root_vel
        mujoco_v_dist_vector = mujoco_vel - mujoco_root_vel
        v_dist = np.linalg.norm(cmu_v_dist_vector - mujoco_v_dist_vector)
        velocity_cost += v_dist

    # Center of mass cost (see DeepMimic page 5, although this implementation differs quite a bit)
    # Only use the Z-axis, at least for now, since it is the easiest to compare
    if use_com_cost:
        com_position_cost = np.linalg.norm(
            mujoco_root_pos[2] - cmu_root_pos[2])
        com_velocity_cost = np.linalg.norm(mujoco_root_vel - cmu_root_vel)
    else:
        com_position_cost = 0
        com_velocity_cost = 0

    # Scale the costs in an attempt to normalize them
    position_cost *= POSITION_COST_SCALE
    velocity_cost *= VELOCITY_COST_SCALE
    com_position_cost *= COM_POSITION_COST_SCALE
    com_velocity_cost *= COM_VELOCITY_COST_SCALE

    # See DeepMimic (p.5) for the full explanation of this calculation
    # This reward is negative and should get closer to zero as it approaches the optimal solution
    total_cost = POSITION_COST_WEIGHT * position_cost \
        + VELOCITY_COST_WEIGHT * velocity_cost \
        + COM_POSITION_COST_WEIGHT * com_position_cost \
        + COM_VELOCITY_COST_WEIGHT * com_velocity_cost
    # return np.exp(-total_cost) * OVERALL_REWARD_SCALE
    return -total_cost * OVERALL_REWARD_SCALE, \
        (POSITION_COST_WEIGHT * position_cost / total_cost,
         VELOCITY_COST_WEIGHT * velocity_cost / total_cost,
         COM_POSITION_COST_WEIGHT * com_position_cost / total_cost,
         COM_VELOCITY_COST_WEIGHT * com_velocity_cost / total_cost)


def get_cmu_position(df, joint_name, frame):
    x = df.values['%s_Xposition' % joint_name][frame]
    # y uses the Zposition because of the weird coordinate system used in .bvh
    y = df.values['%s_Zposition' % joint_name][frame]
    z = df.values['%s_Yposition' % joint_name][frame]
    return np.array([x, y, z])


def get_cmu_velocity(df, joint_name, frame):
    # Use finite-differences to compute the velocity of the given joint
    other_frame = frame - 1 if frame > 0 else frame + 1
    pos1 = get_cmu_position(df, joint_name, frame)
    pos2 = get_cmu_position(df, joint_name, other_frame)
    return (pos1 - pos2) * BVH_RATE  # (f(t + Δt) - f(t)) / Δt


def get_mujoco_position(body, joint_id):
    link_state = p.getLinkState(body, joint_id)
    # In order to match the coordinate system in CMU, flip x and y
    y, x, z = link_state[0]
    return np.array([x, y, z]) * MUJOCO_SCALE_FACTOR


def get_mujoco_velocity(body, joint_id):
    link_state = p.getLinkState(body, joint_id, computeLinkVelocity=True)
    # In order to match the coordinate system in CMU, flip x and y
    v_y, v_x, v_z = link_state[6]
    return np.array([v_x, v_y, v_z]) * MUJOCO_SCALE_FACTOR


def get_and_save_mujoco_joint_id(joints, extremity_dict):
    # Support either providing a joint name or a link name
    joint_to_find = extremity_dict.get('joint_name')
    link_to_find = extremity_dict.get('link_name')
    joint_to_find_id = -1

    for joint in joints:
        # Check if this is a dict of Joint objects, as used by the pybullet gym envs
        if isinstance(joints, dict):
            joint = joints[joint]
            joint_name = joint.joint_name
            joint_index = joint.jointIndex
            link_name = joint.link_name
        else:  # My own definition of joints, used for tests within this file only
            joint_name = joint[0]
            joint_index = joint[1]
            link_name = joint[2]

        if joint_to_find is not None and joint_name == joint_to_find or \
                link_to_find is not None and link_name == link_to_find:
            joint_to_find_id = joint_index
            break

    # Save it for future use
    extremity_dict['joint_id'] = joint_to_find_id
    return joint_to_find_id


def main():
    # Import matplotlib only for this test. Need to import it globally for the alias 'plt' to be available in the draw functions
    globals()['plt'] = __import__('matplotlib.pyplot').pyplot
    from mpl_toolkits.mplot3d import Axes3D

    mujoco_body, mujoco_joints = load_mujoco_file()

    # To test, find the closest and furthest poses in the CMU recording to the mujoco defalut pose
    max_reward = -np.inf
    min_reward = np.inf
    max_frame = -1
    min_frame = -1
    for frame in range(len(cmu_df.values)):
        cost, _ = get_imitation_reward(
            mujoco_body, mujoco_joints, elapsed_time=frame / BVH_RATE)
        if cost > max_reward:
            max_reward = cost
            max_frame = frame
        if cost < min_reward:
            min_reward = cost
            min_frame = frame

    print('Min reward: %s, Max reward: %s' % (min_reward, max_reward))
    print('Min frame: %d, Max frame: %d' % (min_frame, max_frame))

    # Plot the poses, staring with min reward
    ax = draw_stickfigure3d(cmu_df, frame=min_frame, draw_names=DRAW_NAMES)
    draw_stickfigure3d_mujoco(
        mujoco_body, mujoco_joints, ax=ax, draw_names=DRAW_NAMES)

    ax.view_init(10, 40)
    plt.axis('equal')
    plt.show()

    # ... then max reward
    ax = draw_stickfigure3d(cmu_df, frame=max_frame, draw_names=DRAW_NAMES)
    draw_stickfigure3d_mujoco(
        mujoco_body, mujoco_joints, ax=ax, draw_names=DRAW_NAMES)

    ax.view_init(10, 40)
    plt.axis('equal')
    plt.show()


def load_mujoco_file():
    # No need for GUI here
    p.connect(p.DIRECT)
    objs = p.loadMJCF(MUJOCO_XML_FILE,
                      flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

    human = objs[0]
    joints = []

    for j in range(p.getNumJoints(human)):
        info = p.getJointInfo(human, j)

        # Retrieve joints
        # The following could be used to get only joints which are not fixed (can be activated)
        is_fixed = info[2] == p.JOINT_FIXED
        jname = info[1].decode('ascii')
        lname = info[12].decode('ascii')
        # print('joint: %s, link: %s' % (jname, lname))
        parent_index = info[16]
        joints.append((jname, j, lname, parent_index, is_fixed))

    return human, joints


def draw_stickfigure3d_mujoco(body, joints, draw_names=False, ax=None, figsize=(8, 8)):
    SCALE = 16.9  # 15.86
    SHIFT_Y = -20  # -28
    SHIFT_Z = -3.5

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    for joint_name, joint_index, link_name, parent_index, is_fixed in joints:
        # NOTE: Flip x, y in order to match orientation of cmu
        y, x, z = p.getLinkState(body, joint_index)[0]

        # Draw a point on the joint
        ax.scatter(xs=x * SCALE,
                   ys=y * SCALE + SHIFT_Y,
                   zs=z * SCALE + SHIFT_Z,
                   alpha=0.6, c='r', marker='o')

        # Draw a link to the parent
        if parent_index < 0:
            parent_y, parent_x, parent_z = p.getBasePositionAndOrientation(body)[
                0]
        else:
            parent_y, parent_x, parent_z = p.getLinkState(
                body, parent_index)[0]
        # print('%s, idx: %d, parent_idx: %d :: %f -- %f, %f -- %f, %f -- %f' %
        #       (joint_name, joint_index, parent_index, x, parent_x, y, parent_y, z, parent_z))
        ax.plot([parent_x * SCALE, x * SCALE],
                [parent_y * SCALE + SHIFT_Y, y * SCALE + SHIFT_Y],
                [parent_z * SCALE + SHIFT_Z, z * SCALE + SHIFT_Z], 'k-', lw=2)

    for extremity_key in extremities:
        extremity_joint_id = extremities[extremity_key]['mujoco']['joint_id']
        center_of_mass_joint_id = center_of_mass['mujoco']['joint_id']
        root_y, root_x, root_z = p.getLinkState(
            body, center_of_mass_joint_id)[0]
        y, x, z = p.getLinkState(body, extremity_joint_id)[0]

        # Draw joint names over the dots
        if draw_names:
            mjc_dict = extremities[extremity_key]['mujoco']
            joint_name = mjc_dict.get('joint_name')
            link_name = mjc_dict.get('link_name')
            text = joint_name if joint_name is not None else link_name
            ax.text(x=x * SCALE,
                    y=y * SCALE + SHIFT_Y,
                    z=z * SCALE + SHIFT_Z,
                    s=text,
                    color=(0, 0, 0, 0.9))

        # Draw links to extremities
        # See Merel et Al. (Deepmind) on adversarial imitation
        ax.plot([root_x * SCALE, x * SCALE],
                [root_y * SCALE + SHIFT_Y, y * SCALE + SHIFT_Y],
                [root_z * SCALE + SHIFT_Z, z * SCALE + SHIFT_Z], 'g--', lw=1, zorder=2)

    return ax


def draw_stickfigure3d(df, frame, joints=None, draw_names=False, ax=None, figsize=(8, 8)):
    # NOTE This code was originally in https://github.com/omimo/PyMO/blob/5c124c75bee40600cfea9e764431039f33c71ccb/pymo/viz_tools.py
    # but had a bug with the color argument of the ax.text() call. I thus had to copy it here to fix it

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    if joints is None:
        joints_to_draw = df.skeleton.keys()
    else:
        joints_to_draw = joints

    for joint in joints_to_draw:
        parent_x, parent_y, parent_z = get_cmu_position(df, joint, frame)

        ax.scatter(xs=parent_x,
                   ys=parent_y,
                   zs=parent_z,
                   alpha=0.6, c='b', marker='o')

        children_to_draw = [
            c for c in df.skeleton[joint]['children'] if c in joints_to_draw]

        for c in children_to_draw:
            child_x, child_y, child_z = get_cmu_position(df, c, frame)

            ax.plot([parent_x, child_x], [parent_y, child_y], [
                    parent_z, child_z], 'k-', lw=2, c='black')

    for extremity_key in extremities:
        extremity_joint = extremities[extremity_key]['cmu']['joint_name']
        x, y, z = get_cmu_position(df, extremity_joint, frame)

        cmu_center_of_mass_joint = center_of_mass['cmu']['joint_name']
        root_x, root_y, root_z = get_cmu_position(
            df, cmu_center_of_mass_joint, frame)

        # # Draw joint names over the dots
        if draw_names:
            ax.text(x=x + 0.1,
                    y=y + 0.1,
                    z=z + 0.1,
                    s=extremity_joint,
                    color=(0, 0, 0, 0.9))

        # Draw links to extremities
        # See Merel et Al. (Deepmind) on adversarial imitation
        ax.plot([root_x, x],
                [root_y, y],
                [root_z, z], 'g--', lw=1, zorder=2)

    return ax


if __name__ == '__main__':
    main()

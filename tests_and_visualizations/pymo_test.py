from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer
import pymo.viz_tools as vt
from pymo.features import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pybullet as p


# This list is not used anymore. It was used to match joints from the CMU model to joints in the mujoco model
# The jointfix* in the list are also outdated since adding new fixed joints (head and right and left hands)
# mujoco_joint_filter_list = ['jointfix_4_18', 'jointfix_1_11', 'jointfix_8_24', 'jointfix_11_27',
#                             'jointfix_9_22', 'jointfix_10_29', 'jointfix_2_9', 'jointfix_5_16', 'jointfix_6_5']  # TODO
mujoco_link_extremities = ['right_foot', 'left_foot',  # right_foot and left_foot links
                           'right_hand', 'left_hand',  # right_hand and left_hand links
                           'head']  # head link
mujoco_center_of_mass_joint = 'abdomen_x'  # pelvis link was too low
# cmu_joint_list = ['Head', 'Hips', 'RightArm', 'LeftArm', 'RightForeArm', 'LeftForeArm',
#                   'Spine', 'RightUpLeg', 'LeftUpLeg', 'RightLeg', 'LeftLeg', 'RightFoot', 'LeftFoot']  # TODO
cmu_joint_extremities = ['RightHand', 'LeftHand',
                         'RightFoot', 'LeftFoot', 'Head_Nub']  # Head
cmu_center_of_mass_joint = 'Hips'
cmu_joint_list = cmu_joint_extremities + \
    [cmu_center_of_mass_joint]  # Currently only used for showing labels

# HIGHLIGHT
# TODO: Change the .xml file used in the pybullet envs to use this new one instead (added head and hands fixed joints)

# g_frame = 0
# g_interval = 1
# g_forward = True


def main():
    parser = BVHParser()

    parsed_data = parser.parse('walk-02-01-cmu.bvh')
    # parsed_data = parser.parse('walk-01-normal-azumi.bvh')

    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])

    body, joints, center_of_mass_joint_id = init_pybullet()
    compare_extremity_vectors(
        body, joints, center_of_mass_joint_id, positions[0])

    # TODO write the cost function comparing the extremity vectors
    # and, to test it, use it to find the closest and furthest frames from the CMU positions[] data

    # vt.print_skel(parsed_data) # (See output below)
    # vt.draw_stickfigure(positions[0], frame=150)
    ax = draw_stickfigure3d(positions[0], frame=0, draw_names=True)

    # == This is a test to navigate the frames of the CMU recording
    # ax, fig = draw_stickfigure3d(positions[0], frame=g_frame)
    # plt.axis('equal')
    # ax.view_init(10, 40)
    # # plt.show()

    # def onclick(event):
    #     global g_frame
    #     global g_interval
    #     global g_forward
    #     # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #     #       ('double' if event.dblclick else 'single', event.button,
    #     #        event.x, event.y, event.xdata, event.ydata))
    #     if event.button == 1:
    #         g_frame += g_interval if g_forward else -g_interval
    #         plt.cla()
    #         draw_stickfigure3d(positions[0], frame=g_frame, ax=ax)
    #         plt.draw()
    #     elif event.button == 2:
    #         g_forward = not g_forward
    #     elif event.button == 3:
    #         g_interval += 1 if g_forward else -1

    #     print('%s, %s, %r' % (g_frame, g_interval, g_forward))

    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()

    # Find feet ground contact points
    # signal = create_foot_contact_signal(positions[0], 'RightFoot_Yposition')
    # plt.figure(figsize=(12, 6))
    # # plt.plot(signal, 'r')
    # plt.plot(positions[0].values['RightFoot_Yposition'].values, 'g')
    # # plot_foot_up_down(
    # #     positions[0], 'RightFoot_Yposition', t=0.00002, min_dist=100)
    # print(get_foot_contact_idxs(
    #     positions[0].values['RightFoot_Yposition'].values))
    # print(positions[0].values['RightFoot_Yposition'].values[50:100].argmin())

    # peaks = [75, 207]
    # plt.plot(peaks,
    #          positions[0].values['RightFoot_Yposition'].values[peaks], 'ro')
    # plt.show()

    # This is how to get the position of a single joint:
    # Note that positions[0] is a pandas dataframe
    # joint_name = 'Hips'
    # frame = 150
    # print(positions[0].values['%s_Xposition' % joint_name][frame])

    ax = draw_stickfigure3d_mujoco(
        body, joints, center_of_mass_joint_id, draw_names=True, ax=ax)

    plt.axis('equal')
    # ax.view_init(30, 80)
    ax.view_init(0, 0)  # Side view
    # ax.view_init(10, 40)  # Best view imo
    plt.show()
    # while True:
    #     for angle in range(0, 360, 20):
    #         ax.view_init(0, angle) # Use 30 instead of 0 to give the camera an angle down
    #         plt.draw()
    #         plt.pause(1)


def compare_extremity_vectors(body, joints, center_of_mass_joint_id, cmu_df, frame=0):
    cmu_extremity_vectors = []
    mujoco_extremity_vectors = []

    # Get the needed CMU vectors
    root_x = cmu_df.values['%s_Xposition' % cmu_center_of_mass_joint][frame]
    root_y = cmu_df.values['%s_Zposition' % cmu_center_of_mass_joint][frame]
    root_z = cmu_df.values['%s_Yposition' % cmu_center_of_mass_joint][frame]

    for extremity in cmu_joint_extremities:
        extremity_x = cmu_df.values['%s_Xposition' % extremity][frame]
        extremity_y = cmu_df.values['%s_Zposition' % extremity][frame]
        extremity_z = cmu_df.values['%s_Yposition' % extremity][frame]

        vector = (extremity_x - root_x, extremity_y -
                  root_y, extremity_z - root_z)

        cmu_extremity_vectors.append((extremity, vector))

    # Get the needed Mujoco vectors
    root_y, root_x, root_z = p.getLinkState(
        body, center_of_mass_joint_id)[0]

    for extremity in mujoco_link_extremities:
        extremity_joint_id = -1
        for _, joint_index, _, link_name, _ in joints:
            if link_name == extremity:
                extremity_joint_id = joint_index
                break

        extremity_y, extremity_x, extremity_z = p.getLinkState(
            body, extremity_joint_id)[0]

        vector = (extremity_x - root_x, extremity_y -
                  root_y, extremity_z - root_z)

        mujoco_extremity_vectors.append((extremity, vector))

    print("CMU EXTREMITY VECTORS:")
    print(cmu_extremity_vectors)
    print("MUJOCO EXTREMITY VECTORS:")
    print(mujoco_extremity_vectors)

    # TODO pick up from here
    # HIGHLIGHT
    # OUTPUTS:
    # CMU left foot Z: -16.681439731994338
    # MUJOCO left foot Z: -0.8928114104663363
    # SCALE: 18.684169508184524
    # --
    # CMU right foot Z: -16.603285490697278
    # MUJOCO right foot Z: -0.8928114104663363
    # SCALE: 18.596632274250386
    # --
    # CMU head Z: 7.22971318820926
    # MUJOCO head Z: 0.61515867272472
    # SCALE: 11.75259897773483
    # ===
    # CMU head to toe Z: 23.911152920203598
    # MUJOCO head to toe Z: 1.5079700831910563
    # SCALE: 15.856516774924712
    # =-=-=-=-=-=-=-=-=-
    # WITH CMU Head_Nub instead of Head:
    # CMU head nub Z: 8.806852507910357
    # SCALE: 14.316391686233079
    # ===
    # CMU head to toe Z: 25.488292239904695
    # SCALE: 16.90238587888178
    # =-=-=-=-=-=-=-=-=-
    # USING abdomen_x AS ROOT FOR MUJOCO (and Head_Nub):
    # MUJOCO left foot Z: -0.9929682100657364
    # SCALE: 16.79957078473841
    # --
    # MUJOCO head Z: 0.5150018731253199
    # SCALE: 17.10062228408111
    # ===
    # MUJOCO head to toe Z: 1.5079700831910563
    # SCALE: 16.90238587888178


def init_pybullet():
    # No need for GUI here. In fact, is this even necessary?
    p.connect(p.DIRECT)
    objs = p.loadMJCF("isaac/mjcf/humanoid_symmetric_no_ground.xml",  # TODO fix path
                      flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

    human = objs[0]
    joints = []
    center_of_mass_joint_id = -1

    for j in range(p.getNumJoints(human)):
        info = p.getJointInfo(human, j)

        # Retrieve joints
        # The following could be used to get only joints which are not fixed (can be activated)
        is_fixed = info[2] == p.JOINT_FIXED
        jname = info[1].decode('ascii')
        lname = info[12].decode('ascii')
        # print('joint: %s, link: %s' % (jname, lname))
        parent_index = info[16]
        joints.append((jname, j, parent_index, lname, is_fixed))

        # Save the center of mass joint id
        print('jname: %s -- com: %s' % (jname, mujoco_center_of_mass_joint))
        if jname == mujoco_center_of_mass_joint:
            center_of_mass_joint_id = j

    return human, joints, center_of_mass_joint_id


def draw_stickfigure3d_mujoco(body, joints, center_of_mass_joint_id, draw_names=False, ax=None, figsize=(8, 8)):
    SCALE = 16.9  # 15.86
    SHIFT_Y = -20  # -28
    SHIFT_Z = -3.5

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    for joint_name, joint_index, parent_index, link_name, is_fixed in joints:
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

        # Draw joint names over the dots
        if draw_names:
            if link_name in mujoco_link_extremities:
                ax.text(x=x * SCALE,
                        y=y * SCALE + SHIFT_Y,
                        z=z * SCALE + SHIFT_Z,
                        s=link_name,
                        color=(0, 0, 0, 0.9))
            elif joint_name == mujoco_center_of_mass_joint:
                ax.text(x=x * SCALE,
                        y=y * SCALE + SHIFT_Y,
                        z=z * SCALE + SHIFT_Z,
                        s=joint_name,
                        color=(0, 0, 0, 0.9))

        # Draw links to extremities
        # See Merel et Al. (Deepmind) on adversarial imitation
        if link_name in mujoco_link_extremities:
            # NOTE: We could use the base position here, but actually, the pelvis is closer to the actual center of mass
            # root_y, root_x, root_z = p.getBasePositionAndOrientation(body)[0]
            root_y, root_x, root_z = p.getLinkState(
                body, center_of_mass_joint_id)[0]
            ax.plot([root_x * SCALE, x * SCALE],
                    [root_y * SCALE + SHIFT_Y, y * SCALE + SHIFT_Y],
                    [root_z * SCALE + SHIFT_Z, z * SCALE + SHIFT_Z], 'g--', lw=1, zorder=2)

    return ax


def draw_stickfigure3d(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8, 8)):
    # NOTE This code was originally in https://github.com/omimo/PyMO/blob/5c124c75bee40600cfea9e764431039f33c71ccb/pymo/viz_tools.py
    # but had a bug with the color argument of the ax.text() call. I thus had to copy it here to fix it
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints

    if data is None:
        df = mocap_track.values
    else:
        df = data

    for joint in joints_to_draw:
        parent_x = df['%s_Xposition' % joint][frame]
        parent_y = df['%s_Zposition' % joint][frame]
        parent_z = df['%s_Yposition' % joint][frame]
        # ^ In mocaps, Y is the up-right axis

        ax.scatter(xs=parent_x,
                   ys=parent_y,
                   zs=parent_z,
                   alpha=0.6, c='b', marker='o')

        children_to_draw = [
            c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]

        for c in children_to_draw:
            child_x = df['%s_Xposition' % c][frame]
            child_y = df['%s_Zposition' % c][frame]
            child_z = df['%s_Yposition' % c][frame]
            # ^ In mocaps, Y is the up-right axis

            ax.plot([parent_x, child_x], [parent_y, child_y], [
                    parent_z, child_z], 'k-', lw=2, c='black')

        # Draw joint names over the dots
        if draw_names and joint in cmu_joint_list:
            ax.text(x=parent_x + 0.1,
                    y=parent_y + 0.1,
                    z=parent_z + 0.1,
                    s=joint,
                    color=(0, 0, 0, 0.9))

        # Draw links to extremities
        # See Merel et Al. (Deepmind) on adversarial imitation
        if joint in cmu_joint_extremities:
            root_x = df['%s_Xposition' % cmu_center_of_mass_joint][frame]
            root_y = df['%s_Zposition' % cmu_center_of_mass_joint][frame]
            root_z = df['%s_Yposition' % cmu_center_of_mass_joint][frame]
            ax.plot([root_x, parent_x],
                    [root_y, parent_y],
                    [root_z, parent_z], 'g--', lw=1, zorder=2)

    return ax


if __name__ == '__main__':
    main()

# Skeleton (of CMU file)
# NOTE: See 02-01-cmu.png for reference
# What I have discovered is that in mocap files, the captured points are put
# on body joints. Thus, 'RightArm' corresponds, in fact, to the right elbow of the actor
# - Hips (None)
# | | - LowerBack (Hips)
# | | - Spine (LowerBack)
# | | - Spine1 (Spine)
# | | | | - RightShoulder (Spine1)
# | | | | - RightArm (RightShoulder)
# | | | | - RightForeArm (RightArm)
# | | | | - RightHand (RightForeArm)
# | | | | | - RThumb (RightHand)
# | | | | | - RThumb_Nub (RThumb)
# | | | | - RightFingerBase (RightHand)
# | | | | - RightHandIndex1 (RightFingerBase)
# | | | | - RightHandIndex1_Nub (RightHandIndex1)
# | | | - LeftShoulder (Spine1)
# | | | - LeftArm (LeftShoulder)
# | | | - LeftForeArm (LeftArm)
# | | | - LeftHand (LeftForeArm)
# | | | | - LThumb (LeftHand)
# | | | | - LThumb_Nub (LThumb)
# | | | - LeftFingerBase (LeftHand)
# | | | - LeftHandIndex1 (LeftFingerBase)
# | | | - LeftHandIndex1_Nub (LeftHandIndex1)
# | | - Neck (Spine1)
# | | - Neck1 (Neck)
# | | - Head (Neck1)
# | | - Head_Nub (Head)
# | - RHipJoint (Hips)
# | - RightUpLeg (RHipJoint)
# | - RightLeg (RightUpLeg)
# | - RightFoot (RightLeg)
# | - RightToeBase (RightFoot)
# | - RightToeBase_Nub (RightToeBase)
# - LHipJoint (Hips)
# - LeftUpLeg (LHipJoint)
# - LeftLeg (LeftUpLeg)
# - LeftFoot (LeftLeg)
# - LeftToeBase (LeftFoot)
# - LeftToeBase_Nub (LeftToeBase)

# HIGHLIGHT
# Skeleton of mujoco humanoid
# joint: abdomen_z, link: link0_2
# joint: abdomen_y, link: link0_3
# joint: jointfix_7_3, link: lwaist
# joint: abdomen_x, link: link0_5
# joint: jointfix_6_5, link: pelvis
# joint: right_hip_x, link: link0_7
# joint: right_hip_z, link: link0_8
# joint: right_hip_y, link: link0_9
# joint: jointfix_2_9, link: right_thigh
# joint: right_knee, link: link0_11
# joint: jointfix_1_11, link: right_shin
# joint: jointfix_0_10, link: right_foot
# joint: left_hip_x, link: link0_14
# joint: left_hip_z, link: link0_15
# joint: left_hip_y, link: link0_16
# joint: jointfix_5_16, link: left_thigh
# joint: left_knee, link: link0_18
# joint: jointfix_4_18, link: left_shin
# joint: jointfix_3_17, link: left_foot
# joint: right_shoulder1, link: link0_21
# joint: right_shoulder2, link: link0_22
# joint: jointfix_9_22, link: right_upper_arm
# joint: right_elbow, link: link0_24
# joint: jointfix_8_24, link: right_lower_arm
# joint: left_shoulder1, link: link0_26
# joint: left_shoulder2, link: link0_27
# joint: jointfix_11_27, link: left_upper_arm
# joint: left_elbow, link: link0_29
# joint: jointfix_10_29, link: left_lower_arm

# HIGHLIGHT updated list after adding headand left and right hand fixed joints
# joint: jointfix_0_0, link: head
# joint: abdomen_z, link: link0_3
# joint: abdomen_y, link: link0_4
# joint: jointfix_8_4, link: lwaist
# joint: abdomen_x, link: link0_6
# joint: jointfix_7_6, link: pelvis
# joint: right_hip_x, link: link0_8
# joint: right_hip_z, link: link0_9
# joint: right_hip_y, link: link0_10
# joint: jointfix_3_10, link: right_thigh
# joint: right_knee, link: link0_12
# joint: jointfix_2_12, link: right_shin
# joint: jointfix_1_11, link: right_foot
# joint: left_hip_x, link: link0_15
# joint: left_hip_z, link: link0_16
# joint: left_hip_y, link: link0_17
# joint: jointfix_6_17, link: left_thigh
# joint: left_knee, link: link0_19
# joint: jointfix_5_19, link: left_shin
# joint: jointfix_4_18, link: left_foot
# joint: right_shoulder1, link: link0_22
# joint: right_shoulder2, link: link0_23
# joint: jointfix_11_23, link: right_upper_arm
# joint: right_elbow, link: link0_25
# joint: jointfix_10_25, link: right_lower_arm
# joint: jointfix_9_24, link: right_hand
# joint: left_shoulder1, link: link0_28
# joint: left_shoulder2, link: link0_29
# joint: jointfix_14_29, link: left_upper_arm
# joint: left_elbow, link: link0_31
# joint: jointfix_13_31, link: left_lower_arm
# joint: jointfix_12_30, link: left_hand

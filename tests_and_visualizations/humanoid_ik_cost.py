import pybullet as p
import pybullet_data
import time
import os
from typing import List, Dict, Tuple

import imitation_reward
from gym_env.robot_bases import Joint

p.connect(p.GUI)
p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))
# load URDF, given a relative or absolute file+path
obUids = p.loadMJCF(os.path.join(
    "res", "humanoid_symmetric_no_ground.xml"))
humanoid = obUids[0]

jointIds: List[int] = []
paramIds: List[int] = []
joints: Dict[str, Joint] = {}

TIME_STEP = 0.0165
elapsed_time = 0.

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)
p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(1)

# Disable the GUI
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

xPosId = p.addUserDebugParameter('X Position', -5, 5, 0)
yPosId = p.addUserDebugParameter('Y Position', -5, 5, 0)
zPosId = p.addUserDebugParameter('Z Position', 0, 3, 1.2)

for j in range(p.getNumJoints(humanoid)):
    p.changeDynamics(humanoid, j, mass=0, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(humanoid, j)
    # print(info)
    jointName = info[1].decode("utf8")
    jointType = info[2]
    partName = info[12].decode("utf8")
    joints[jointName] = Joint(p, jointName, partName, obUids, 0, j)
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)
        paramIds.append(p.addUserDebugParameter(
            '%d: %s' % (len(jointIds) - 1, jointName), -4, 4, 0))

jointPoses = [(0, 0, 0)]*17
extremities_to_update = ['left_hand', 'right_hand',
                         'left_foot', 'right_foot', 'head']
extremities_indices = [[14, 15, 16], [11, 12, 13],
                       [7, 8, 9, 10], [3, 4, 5, 6], [0, 1, 2]]

# Multiple sphere bodies can be created from this one visual shape
visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=0.08)
visual_shapes = []
for i in range(len(extremities_to_update)):
    visual_shapes.append(p.createMultiBody(baseVisualShapeIndex=visualShapeId))

while(1):
    xPos = p.readUserDebugParameter(xPosId)
    yPos = p.readUserDebugParameter(yPosId)
    zPos = p.readUserDebugParameter(zPosId)
    p.resetBasePositionAndOrientation(
        humanoid, (xPos, yPos, zPos), (0, 0, 0, 1))

    frame = int(elapsed_time *
                imitation_reward.BVH_RATE) % imitation_reward.frame_count

    # Get the center of mass position
    cmu_center_of_mass_joint = imitation_reward.center_of_mass['cmu']['joint_name']
    rootY, rootX, _ = imitation_reward.get_cmu_position(
        imitation_reward.cmu_df, cmu_center_of_mass_joint, frame) / imitation_reward.MUJOCO_SCALE_FACTOR

    # FIXME The arms are messed up, seem to be facing backwards for some reason...
    # Using a nullspace IK may solve this issue

    # Pybullet does not have inverse kinematics with multiple end effectors,
    # so do multiple inverse kinematics computations for each extremity
    for extremity_to_update, extremity_indices, visual_shape in zip(extremities_to_update, extremities_indices, visual_shapes):
        endEffectorLinkIndex = imitation_reward.get_and_save_mujoco_joint_id(
            joints, imitation_reward.extremities[extremity_to_update]['mujoco'])
        targetPosY, targetPosX, targetPosZ = imitation_reward.get_cmu_position(
            imitation_reward.cmu_df, imitation_reward.extremities[extremity_to_update]['cmu']['joint_name'], frame) / imitation_reward.MUJOCO_SCALE_FACTOR
        targetPos = (targetPosX - rootX, targetPosY - rootY, targetPosZ)
        extremityJointPoses = p.calculateInverseKinematics(
            humanoid, endEffectorLinkIndex, targetPos)

        for i in extremity_indices:
            jointPoses[i] = extremityJointPoses[i]

        # Show the actual extremity positions in order to debug
        p.resetBasePositionAndOrientation(
            visual_shape, targetPos, (0, 0, 0, 1))

    for i in range(len(paramIds)):
        c = paramIds[i]
        # targetPos = p.readUserDebugParameter(c) # Use user-defined positions
        targetPos = jointPoses[i]  # Use inverse kinematics positions
        # p.setJointMotorControl2(
        #     humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5*240.)
        p.resetJointState(humanoid, jointIds[i], targetPos)

    (reward, _), phase = imitation_reward.get_imitation_reward_and_phase(
        humanoid, joints, elapsed_time=elapsed_time, use_com_cost=False)

    print('%.2f: %.4f' % (phase, reward))

    elapsed_time += TIME_STEP
    time.sleep(TIME_STEP)

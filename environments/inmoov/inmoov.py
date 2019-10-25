import math
import os

import numpy as np
import pybullet as p
import pybullet_data
# from ipdb import set_trace as tt

from environments.inmoov.joints_registry import joint_registry
URDF_PATH = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot/"
GRAVITY = -9.8


class Inmoov:
    def __init__(self, urdf_path=URDF_PATH):
        self.urdf_path = urdf_path
        self._renders = True
        self.debug_mode = True
        self.inmoov_id = -1
        self.num_joints = -1
        self.robot_base_pos = [0, 0, 0]


        if self.debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])

            # To debug the joints of the Inmoov robot
            debug_joints = []
            self.joints_key = []
            for joint_index in joint_registry:
                self.joints_key.append(joint_index)
                debug_joints.append(p.addUserDebugParameter(joint_registry[joint_index], -1., 1., 0))
            self.debug_joints = debug_joints

            # To debug the camera position
            debug_camera = 0
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        p.setGravity(0., 0., -10.)
        self.inmoov_id = p.loadURDF(os.path.join(self.urdf_path, 'inmoov_col.urdf'), self.robot_base_pos)
        self.num_joints = p.getNumJoints(self.inmoov_id)
        # tmp1 = p.getNumBodies(self.inmoov_id)  # Equal to 1, only one body
        # tmp2 = p.getNumConstraints(self.inmoov_id)  # Equal to 0, no constraint?
        # tmp3 = p.getBodyUniqueId(self.inmoov_id)  # res = 0, do not understand
        for jointIndex in range(self.num_joints):
            p.resetJointState(self.inmoov_id, jointIndex, 0.1 )

        # tt( )

    def debugger_step(self):

        if self.debug_mode:
            current_joints = []
            # The order is as the same as the self.joint_key
            for j in self.debug_joints:
                tmp_joint_control = p.readUserDebugParameter(j)
                current_joints.append(tmp_joint_control)
            for joint_state, joint_key in zip(current_joints, self.joints_key):
                p.resetJointState(self.inmoov_id, joint_key, targetValue=joint_state)
            p.stepSimulation()

    def debugger_camera(self):
        if self.debug_mode:
            tete = "Stupid"

import time
if __name__ == '__main__':

    robot = Inmoov()
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # robot = Inmoov()
    for i in range(1000000):
        time.sleep(0.01)
        robot.debugger_step()

    p.disconnect()

import math
import os

import numpy as np
import pybullet as p
import pybullet_data
from ipdb import set_trace as tt

from environments.inmoov.joints_registry import joint_registry, control_joint
URDF_PATH = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot/"
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 128,128

class Inmoov:
    def __init__(self, urdf_path=URDF_PATH):
        self.urdf_path = urdf_path
        self._renders = True
        self.debug_mode = True
        self.inmoov_id = -1
        self.num_joints = -1
        self.robot_base_pos = [0, 0, 0]
        # constraint
        self.max_force = 200.
        self.max_velocity = .35

        # camera position
        self.camera_target_pos = (0.316, -0.2, -0.1)
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

    def apply_action(self, motor_commands):
        """
        Apply the action to the inmoov robot joint
        :param motor_commands:
        """
        joint_poses = motor_commands

        # TODO: i is what?
        tt()
        p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force,
                                maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)


    def step(self, action):
        assert len(action) == len(control_joint)
        # TODO
        return

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

    def render(self):
        camera_target_position = self.camera_target_pos
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            camera_target_position=camera_target_position,
            distrance=1.,
            yaw=145,
            pitch=0,
            roll=0,
            upAxisIndex=2
        )

        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)

        p.getCameraImage(
            width=128, height=128, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)


import time
if __name__ == '__main__':

    robot = Inmoov()
    _urdf_path = pybullet_data.getDataPath()
    # planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    stadiumId = p.loadSDF(os.path.join(_urdf_path, "stadium.sdf"))
    sjtu_urdf_path = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot"
    #tomato1Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0,1,0.5] )
    tomato2Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5], baseOrientation=[0,0,0,1])
    # robot = Inmoov()
    while True:
        time.sleep(0.01)
        robot.debugger_step()
        robot.apply_action([1,2,3])

    p.disconnect()

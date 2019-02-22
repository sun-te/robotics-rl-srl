from __future__ import division, print_function, absolute_import
from gym import spaces
from gym import logger
import numpy as np
import cv2

from real_robots.constants import *


def velocity2pos(robot, speed_x, speed_y, speed_yaw):
    """
    Calculate the robot position
    Assume in 1/RL_CONTROL_FREQ, the heading remains the same (not true,
    but should be approximately work if RL_CONTROL_FREQ is high enough)
    translate the last velocity cmd in robot local coordiante to position cmd in gound coordiante

    :param robot: agent that is performing the continuous actions
    :param speed_x: (float)
    :param speed_y: (float)
    :param speed_yaw: (float)
    :return: (float, float, float) final position (x, y, yam) of the robot as after performing the action
    """
    cos_direction = np.cos(robot.robot_yaw)
    sin_direction = np.sin(robot.robot_yaw)

    ground_pos_cmd_x = robot.robot_pos[0] + (speed_x * cos_direction - speed_y * sin_direction) / RL_CONTROL_FREQ
    ground_pos_cmd_y = robot.robot_pos[1] + (speed_y * cos_direction + speed_x * sin_direction) / RL_CONTROL_FREQ
    ground_yaw_cmd = robot.robot_yaw + speed_yaw / RL_CONTROL_FREQ
    return ground_pos_cmd_x, ground_pos_cmd_y, ground_yaw_cmd


def wheelSpeed2pos(robot, left_speed, front_speed, right_speed):
    """
    Calculate the robot position by omnirobot's kinematic equations
    Assume in 1/RL_CONTROL_FREQ, the heading remains the same (not true,
    but should be approximately work if RL_CONTROL_FREQ is high enough)

    :param robot: agent that is performing the continuous actions
    :param left_speed: (float)
    :param front_speed: (float)
    :param right_speed: (float)
    :return: (float, float, float) final position (x, y, yam) of the robot as after performing the action
    """

    # translate the last wheel speeds cmd in last velocity cmd
    local_speed_x = left_speed / np.sqrt(3.0) - right_speed / np.sqrt(3.0)
    local_speed_y = - front_speed / 1.5 + left_speed / 3.0 + right_speed / 3.0
    local_rot_speed = - front_speed / (3.0 * OMNIROBOT_L) - left_speed / (3.0 * OMNIROBOT_L) \
                      - right_speed / (3.0 * OMNIROBOT_L)

    # translate the last velocity cmd in robot local coordiante to position cmd in gound coordiante
    cos_direction = np.cos(robot.robot_yaw)
    sin_direction = np.sin(robot.robot_yaw)

    ground_pos_cmd_x = robot.robot_pos[0] + (local_speed_x *
                                            cos_direction - local_speed_y * sin_direction) / RL_CONTROL_FREQ
    ground_pos_cmd_y = robot.robot_pos[1] + (local_speed_y *
                                            cos_direction + local_speed_x * sin_direction) / RL_CONTROL_FREQ
    ground_yaw_cmd = robot.robot_yaw + local_rot_speed / RL_CONTROL_FREQ
    return ground_pos_cmd_x, ground_pos_cmd_y, ground_yaw_cmd


class PosTransformer(object):
    def __init__(self, camera_mat: np.ndarray, dist_coeffs: np.ndarray,
                 pos_camera_coord_ground: np.ndarray, rot_mat_camera_coord_ground: np.ndarray):
        """
        Transform the position among physical position in camera coordinate,
                                     physical position in ground coordinate,
                                     pixel position of image
        """
        super(PosTransformer, self).__init__()
        self.camera_mat = camera_mat

        self.dist_coeffs = dist_coeffs

        self.camera_2_ground_trans = np.zeros((4, 4), np.float)
        self.camera_2_ground_trans[0:3, 0:3] = rot_mat_camera_coord_ground
        self.camera_2_ground_trans[0:3, 3] = pos_camera_coord_ground
        self.camera_2_ground_trans[3, 3] = 1.0

        self.ground_2_camera_trans = np.linalg.inv(self.camera_2_ground_trans)

    def phyPosCam2PhyPosGround(self, pos_coord_cam):
        """
        Transform physical position in camera coordinate to physical position in ground coordinate
        """
        assert pos_coord_cam.shape == (3, 1)
        homo_pos = np.ones((4, 1), np.float32)
        homo_pos[0:3, :] = pos_coord_cam
        return (np.matmul(self.camera_2_ground_trans, homo_pos))[0:3, :]

    def phyPosGround2PixelPos(self, pos_coord_ground, return_distort_image_pos=False):
        """
        Transform the physical position in ground coordinate to pixel position
        """
        pos_coord_ground = np.array(pos_coord_ground)
        if len(pos_coord_ground.shape) == 1:
            pos_coord_ground = pos_coord_ground.reshape(-1, 1)

        assert pos_coord_ground.shape == (
            3, 1) or pos_coord_ground.shape == (2, 1)

        homo_pos = np.ones((4, 1), np.float32)
        if pos_coord_ground.shape == (2, 1):
            # by default, z =0 since it's on the ground
            homo_pos[0:2, :] = pos_coord_ground
            
            # (np.random.randn() - 0.5) * 0.05 # add noise to the z-axis
            homo_pos[2, :] = 0
        else:
            homo_pos[0:3, :] = pos_coord_ground
        homo_pos = np.matmul(self.ground_2_camera_trans, homo_pos)
        
        pixel_points, _ = cv2.projectPoints(homo_pos[0:3, :].reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            self.camera_mat, self.dist_coeffs if return_distort_image_pos else None)
        return pixel_points.reshape((2, 1))


class BiggerBox(spaces.Box):
    """
    A N dimentional space with all coordinates bounded.
    """

    def __init__(self, limits=None, shape=None, dtype=None):

        super(BiggerBox, self).__init__(low=0.0, high=0.0, shape=shape, dtype=dtype)

        if shape is None:
            for x in limits:
                assert all([y.shape == x.shape for y in limits])
            self.shape = limits[0].shape
        else:
            assert all([np.isscalar(x) for x in limits])
            for i in range(len(limits)):
                limits[i] += 0.0
            self.shape = shape

        if dtype is None:  # Autodetect type
            if (limits[0] == 255).all():
                self.dtype = np.uint8
            else:
                self.dtype = np.float32
            logger.warn(
                "Ring Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
        else:
            self.dtype = dtype
        self.limits = [x.astype(self.dtype) for x in limits]
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return np.array([self.np_random.uniform(low=-x, high=x) for x in self.limits]).astype(self.dtype)

    def contains(self, action):
        return action.shape == self.shape and \
               all([np.logical_and(action[i] >= -self.limits[i],
                                   action[i] <= self.limits[i]) for i in range(len(action))])

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "BiggerBox" + str(self.shape)

    def __eq__(self, other):
        return all([np.allclose(self.limits[i] , other.limits[i]) for i in range(len(self.limits))])

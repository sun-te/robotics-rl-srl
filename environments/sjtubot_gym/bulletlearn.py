import pybullet as p
import time
import pybullet_data
from termcolor import colored
import os
from ipdb import set_trace as tt
def printGreen(string):
    """
    Print a string in green in the terminal
    :param string: (str)
    """
    print(colored(string, 'green'))


def printYellow(string):
    """
    :param string: (str)
    """
    print(colored(string, 'yellow'))


def printRed(string):
    """
    :param string: (str)
    """
    print(colored(string, 'red'))


def printBlue(string):
    """
    :param string: (str)
    """
    print(colored(string, 'blue'))


_urdf_path = pybullet_data.getDataPath()
custom_urdf_path = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf"
sjtu_urdf_path = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot"
pybullet_data_path = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/pybullet_data"
# physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

#p.setAdditionalSearchPath('urdf') #optionally
p.setGravity(0,0,-10)

planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))

cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# r2d2Id = p.loadURDF(os.path.join(_urdf_path, "r2d2.urdf"), cubeStartPos, cubeStartOrientation)
# gripperId = p.loadURDF(os.path.join(pybullet_data_path, "gripper/wsg50_one_motor_gripper_left_finger.urdf"), [0,0,0])
# huskyId = p.loadURDF(os.path.join(pybullet_data_path, "husky/husky.urdf"), [0,0,0])
# jengaId = p.loadURDF(os.path.join(pybullet_data_path, "jenga/jenga.urdf"), [0,0,0])
# kuka = p.loadURDF(os.path.join(pybullet_data_path, "kuka_iiwa/model_vr_limits.urdf"), [0,0,0])
# lego = p.loadURDF(os.path.join(pybullet_data_path, "lego/lego.urdf"), [0,0,0])
# quadruped1 = p.loadURDF(os.path.join(pybullet_data_path, "quadruped/minitaur.urdf"), [0,0,0])
# quadruped = p.loadURDF(os.path.join(pybullet_data_path, "quadruped/minitaur_fixed_all.urdf"), [0,0,0])
# racer = p.loadURDF(os.path.join(pybullet_data_path, "racecar/racecar.urdf"), [0,0,0])
# table = p.loadURDF(os.path.join(pybullet_data_path, "table_square/table_squre.urdf"), [0,0,0])
# tray = p.loadURDF(os.path.join(pybullet_data_path, "tray/tray_textured2.urdf"), [0,0,0])
# object = p.loadURDF(os.path.join(pybullet_data_path, "teddy_vhacd.urdf"), [0,0,0])
# object2 = p.loadSDF(os.path.join(pybullet_data_path, "stadium.sdf"))
# objects = p.loadSDF(os.path.join(_urdf_path, "kuka_iiwa/kuka_with_gripper2.sdf"))
# sjtuID = p.loadURDF(os.path.join(sjtu_urdf_path,"inmoov_right_hand.urdf"), [1,1,1] )
sjtuID = p.loadURDF(os.path.join(sjtu_urdf_path,"inmoov_col.urdf"), [0,0,0] )
# modelId = p.loadURDF(os.path.join(pybullet_data_path, "kuka_iiwa/model_free_base.urdf"),[1,1,1])


for i in range (1000000):
    p.stepSimulation()
time.sleep(1.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#printRed("{} {}".format(cubePos,cubeOrn))
p.disconnect()

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
# physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

#p.setAdditionalSearchPath('urdf') #optionally
p.setGravity(0,0,0)

planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))

cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF(os.path.join(_urdf_path, "r2d2.urdf"), cubeStartPos, cubeStartOrientation)
# modelId = p.loadURDF(os.path.join(_urdf_path, "kuka_iiwa/model_vr_limits.urdf"), [1,1,1])
# objects = p.loadSDF(os.path.join(_urdf_path, "kuka_iiwa/kuka_with_gripper2.sdf"))
sjtuID = p.loadURDF(os.path.join(sjtu_urdf_path,"inmoov.urdf"), [1,1,1] )
for i in range (10000):
    p.stepSimulation()
time.sleep(1.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
printRed("{} {}".format(cubePos,cubeOrn))
p.disconnect()

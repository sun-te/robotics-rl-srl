import pybullet as p
import time
import pybullet_data
from termcolor import colored
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
# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
racerId = p.loadURDF("racecar/racecar.urdf", [1,1,1], cubeStartOrientation)

for i in range (10000):
    p.stepSimulation()
time.sleep(1.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
printRed("{} {}".format(cubePos,cubeOrn))
p.disconnect()

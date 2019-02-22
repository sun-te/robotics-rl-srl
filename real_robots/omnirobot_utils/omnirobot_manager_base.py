from __future__ import division, print_function, absolute_import

from real_robots.constants import *
from real_robots.omnirobot_utils.utils import wheelSpeed2pos, velocity2pos


class OmnirobotManagerBase(object):
    def __init__(self, second_cam_topic=None):
        """
        This class is the basic class for omnirobot server, and omnirobot simulator's server.
        This class takes omnirobot position at instant t, and takes the action at instant t,
        to determinate the position it should go at instant t+1, and the immediate reward it can get at instant t
        """
        super(OmnirobotManagerBase, self).__init__()
        self.second_cam_topic = SECOND_CAM_TOPIC
        self.episode_idx = 0

        # the abstract object for robot,
        # can be the real robot (Omnirobot class)
        #  or the robot simulator (OmniRobotEnvRender class)
        self.robot = None

    def rightAction(self):
        """
        Let robot excute right action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] - STEP_DISTANCE > MIN_Y:
            self.robot.right()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def leftAction(self):
        """
        Let robot excute left action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] + STEP_DISTANCE < MAX_Y:
            self.robot.left()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def forwardAction(self):
        """
        Let robot excute forward action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] + STEP_DISTANCE < MAX_X:
            self.robot.forward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def backwardAction(self):
        """
        Let robot excute backward action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] - STEP_DISTANCE > MIN_X:
            self.robot.backward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def moveContinousAction(self, msg):
        """
        Let robot excute continous action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if MIN_X < self.robot.robot_pos[0] + msg['action'][0] < MAX_X and \
                MIN_Y < self.robot.robot_pos[1] + msg['action'][1] < MAX_Y:
            self.robot.moveContinous(msg['action'])
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def moveByLinearAccCmd(self, msg):
        """
        TODO: constraints ?
        :param msg: (float, float, float) as action to perform (acc_x, acc_y, acc_yaw) (m/s^2,m/s^2,rad/s^2)
                    which is the acceleration of robot's velocity, presented in the robot's local frame
        :return: (bool) Whether or not did the robot bump into the wall
        """
        acc_x, acc_y = msg['action'][0], msg['action'][1]
        acc_yaw = msg['action'][2] if len(msg['action']) >= 3 else 0
        print("acc_yaw", acc_yaw)
        speed_x = self.robot.curr_robot_velocity[0] + acc_x / RL_CONTROL_FREQ
        speed_y = self.robot.curr_robot_velocity[1] + acc_y / RL_CONTROL_FREQ
        speed_yaw = self.robot.curr_robot_velocity[2] + acc_yaw / RL_CONTROL_FREQ
        print("speed_yaw",speed_yaw)
        ground_pos_cmd_x, ground_pos_cmd_y, _ = velocity2pos(self.robot, speed_x, speed_y, speed_yaw)
        print("ground_pos_cmd_x",ground_pos_cmd_x)
        print('ground_pos_cmd_y',ground_pos_cmd_y)
        if MIN_X < ground_pos_cmd_x < MAX_X and MIN_Y < ground_pos_cmd_y < MAX_Y:
            self.robot.moveByVelocityCmd(speed_x, speed_y, speed_yaw)
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def moveByWheelsAccCmd(self, msg):
        """
        :param msg: (float, float, float) as action to perform(acc_speed, acc_speed, acc_speed),
                    which is the acceleration of wheels' linear speed.  (m/s^2,m/s^2,m/s^2)
        :return: (bool) Whether or not did the robot bump into the wall
        """
        acc_left, acc_front, acc_right = msg['action']
        left_speed = self.robot.curr_wheel_speeds[0] + acc_left / RL_CONTROL_FREQ
        front_speed = self.robot.curr_wheel_speeds[1] + acc_front / RL_CONTROL_FREQ
        right_speed = self.robot.curr_wheel_speeds[2] + acc_right / RL_CONTROL_FREQ
        print(acc_left, acc_front, acc_right)
        ground_pos_cmd_x, ground_pos_cmd_y, _ = wheelSpeed2pos(self.robot, left_speed, front_speed, right_speed)

        if MIN_X < ground_pos_cmd_x < MAX_X and MIN_Y < ground_pos_cmd_y < MAX_Y:
            self.robot.moveByWheelsCmd(left_speed, front_speed, right_speed)
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def sampleRobotInitalPosition(self):
        random_init_x = np.random.random_sample() * (INIT_MAX_X - INIT_MIN_X) + INIT_MIN_X
        random_init_y = np.random.random_sample() * (INIT_MAX_Y - INIT_MIN_Y) + INIT_MIN_Y
        return [random_init_x, random_init_y]
    
    def resetEpisode(self):
        """
        Give the correct sequance of commands to the robot 
        to rest environment between the different episodes
        """
        if self.second_cam_topic is not None:
            assert NotImplementedError
        # Env reset
        random_init_position = self.sampleRobotInitalPosition()
        self.robot.setRobotCmd(random_init_position[0], random_init_position[1], 0)

    def processMsg(self, msg):
        """
        Using this steps' msg command the determinate the correct position that the robot should be at next step,
        and to determinate the reward of this step.
        This function also takes care of the environment's reset.
        :param msg: (dict)
        """
        command = msg.get('command', '')
        if command == 'reset':
            action = None
            self.episode_idx += 1
            self.resetEpisode()

        elif command == 'action':
            if msg.get('is_discrete', False):
                action = Move(msg['action'])
            else:
                action = 'Continuous'

        elif command == "exit":
            print("recive exit signal, quit...")
            exit(0)
        else:
            raise ValueError("Unknown command: {}".format(msg))

        has_bumped = False
        # We are always facing North
        if action == Move.FORWARD:
            has_bumped = self.forwardAction()
        elif action == Move.STOP:
            pass
        elif action == Move.RIGHT:
            has_bumped = self.rightAction()
        elif action == Move.LEFT:
            has_bumped = self.leftAction()
        elif action == Move.BACKWARD:
            has_bumped = self.backwardAction()
        elif action == 'Continuous':
            if msg['use_position']:
                has_bumped = self.moveContinousAction(msg)
            elif msg['use_linear_acceleration']:
                has_bumped = self.moveByLinearAccCmd(msg)
            elif msg['use_wheel_acceleration']:
                has_bumped = self.moveByWheelsAccCmd(msg)
            else:
                pass

        elif action == None:
            pass
        else:
            print("Unsupported action: ", action)

        # Determinate the reward for this step
        
        # Consider that we reached the target if we are close enough
        # we detect that computing the difference in area between TARGET_INITIAL_AREA
        # current detected area of the target
        if np.linalg.norm(np.array(self.robot.robot_pos) - np.array(self.robot.target_pos)) \
                < DIST_TO_TARGET_THRESHOLD:
            self.reward = REWARD_TARGET_REACH
        elif has_bumped:
            self.reward = REWARD_BUMP_WALL
        else:
            self.reward = REWARD_NOTHING

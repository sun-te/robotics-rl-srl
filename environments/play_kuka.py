import argparse
import cv2
import numpy as np
# import time

from environments.registry import registered_env
# from environments.utils import makeEnv
from ipdb import set_trace as tt


def discrete_step_plot(env, name="Kuka arm"):
    """
    # action description:
    # 0 for -x, 1 for +x
    # 2 for -y, 3 for +y
    # 4 for -z, 5 for +z
    :return: an action
    """
    img = env.reset()[...,::-1]
    #action = input("Please enter an action (from 0-5): ")
    # since opencv use bgr to plot
    # img = img[...,::-1]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 512, 512)
    cv2.imshow(name, img)
    while True:
        # np.array(self._observation), reward, done, {}
        k = cv2.waitKey(10000) - 48
        print(k)
        if k == -21:
            cv2.destroyAllWindows()
            break
        elif k >= 0 and k < 6:
            action = k
            img = env.step(action)[0][...,::-1]
            cv2.imshow(name, img)
        else:
            cv2.imshow(name, img)
            print("Input is not valid or no input, number between 0-5, press ESC to quit")


def main():
    parser = argparse.ArgumentParser(description='Play Kuka environement')
    parser.add_argument('--env', type=str, default='KukaButtonGymEnv-v0')
    parser.add_argument('--num-cpu', type=int, default=1, help='number of cpu to run on')
    parser.add_argument('--num-episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('--name', type=str, default='kuka_button', help='Folder name for the output')
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no-record-data', action='store_true', default=False)
    parser.add_argument('--max-distance', type=float, default=0.28,
                        help='Beyond this distance from the goal, the agent gets a negative reward')
    parser.add_argument('-c', '--continuous-actions', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('-r', '--random-target', action='store_true', default=False,
                        help='Set the button to a random position')
    parser.add_argument('--force-down', action='store_true', default=False,
                        help='Whether to force the robot arm to force going down')
    parser.add_argument('--multi-view', action='store_true', default=False, help='Set a second camera to the scene')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--reward-dist', action='store_true', default=False,
                        help='Prints out the reward distribution when the dataset generation is finished')
    parser.add_argument('--episode', type=int, default=-1,
                        help='Model saved at episode N that we want to load')
    args = parser.parse_args()
    env_class = registered_env[args.env][0]
    env = env_class(force_down=args.force_down, multi_view=args.multi_view)

    discrete_step_plot(env)



if __name__ == '__main__':
    main()

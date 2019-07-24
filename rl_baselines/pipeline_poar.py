"""
baseline benchmark script for openAI RL Baselines
"""
import os
import argparse
import subprocess

import yaml
import numpy as np

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from state_representation import SRLType
from srl_zoo.utils import printGreen, printRed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines Benchmark",
                                     epilog='After the arguments are parsed, the rest are assumed to be arguments for' +
                                            ' rl_baselines.train')
    parser.add_argument('--algo', type=str, default='poar', help='OpenAI baseline to use',
                        choices=list(registered_rl.keys()))
    parser.add_argument('--env', type=str, nargs='+', default=["MobileRobotGymEnv-v0"], help='environment ID(s)',
                        choices=list(registered_env.keys()))
    parser.add_argument('--timesteps', type=int, default=2e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--num-iteration', type=int, default=15,
                        help='number of time each algorithm should be run for each unique combination of environment ' +
                             ' and srl-model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='initial seed for each unique combination of environment and srl-model.')
    parser.add_argument('--gpu', type=str, default='0')

    # returns the parsed arguments, and the rest are assumed to be arguments for rl_baselines.train
    args, train_args = parser.parse_known_args()
    envs = args.env
    seeds = np.arange(args.num_iteration)

    printGreen("\nRunning {} benchmarks {} times...".format(args.algo, args.num_iteration))
    print("environments:\t{}".format(envs))
    print("verbose:\t{}".format(args.verbose))
    print("timesteps:\t{}".format(args.timesteps))


    # 'reconstruction, forward, inverse, state_entropy, reward'
    srl_weights = [
                   [10,1,1,1,1],
                   [5, 1,1,1,1],
                   [1, 1, 1, 1, 1],
                   [5,1,1,0,0],
                   [1, 1, 1, 0, 0],
                   [0,0,0,0,0],
                   [1,0,1,0,1], # ae + inverse +reward
                   [0,1,1,1,0],   # f +i + entropy
                   [1,0,1,0,100]]
    srl_name = ['a','f','i','e','r']
    counter = 0
    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        # shut the g** d** mouth
        stdout = open(os.devnull, 'w')

    for i in range(1, args.num_iteration+1):
        for env in envs:
            printGreen(
                "\nIteration_num={} (seed: {}), Environment='{}', Algo='{}'".format(i, seeds[i-1], env, args.algo))
            for weights in srl_weights:

                log_dir = 'logs/POAR/srl_'
                weight_args = ['--srl-weight']
                for j, w in enumerate(weights):
                    if w > 0:
                        log_dir += srl_name[j]+str(w)
                    weight_args += [str(w)]
                log_dir += '/'

                loop_args = ['--seed', str(seeds[i-1]), '--algo', args.algo, '--env', env, '--srl-model', 'raw_pixels',
                             '--num-timesteps', str(int(args.timesteps)),
                             '--log-dir', log_dir, '--gpu', str(args.gpu), '--num-cpu', '4',
                             '-r']
                loop_args += weight_args
                poar_args = ['--state-dim', '200','--structure', 'srl_autoencoder']
                loop_args += poar_args
                subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)


if __name__ == '__main__':
    main()

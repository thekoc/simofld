"""To plot the result.
"""

import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from simofld.masl import CloudServer, MobileUser, RayleighChannel


from simofld.envs import create_env
STEP = 10
if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = max(Path('.').glob(r'results-*')).as_posix()
    with open(filename) as f:
        results = json.load(f)
    gamma_results = [r for r in results if r[0]['group'] == 'gamma']
    lr_results = [r for r in results if r[0]['group'] == 'lr']
    user_num_results = [r for r in results if r[0]['group'] == 'user_num']
    channel_num_results = [r for r in results if r[0]['group'] == 'channel_num']

    if gamma_results:
        fig, ax = plt.subplots()
        for r in gamma_results:
            step = STEP
            gamma = r[0]['gamma']
            cost = r[1]['system_cost_histogram']
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')

            ax.plot(Y, label=f'gamma: {gamma:e}')
            # ax.set_ylim(bottom=0)
            ax.legend()
    
    if lr_results:
        fig, ax = plt.subplots()
        for r in lr_results:
            step = STEP
            lr = r[0]['lr']
            cost = r[1]['system_cost_histogram']
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')

            ax.plot(Y, label=f'lr: {lr:e}')
            # ax.set_ylim(bottom=0)
            ax.legend()

    if user_num_results:
        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in user_num_results:
            step = STEP
            X.append(r[0]['user_num'])
            Y.append(r[1]['final_system_cost'])
        ax.set_ylabel('System wide cost')
        ax.set_xlabel('User number')
        ax.plot(X, Y)
        ax.legend()

        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in user_num_results:
            step = STEP
            X.append(r[0]['user_num'])
            Y.append(r[1]['final_beneficial_user_num'])
        ax.set_ylabel('Beneficial user')
        ax.set_xlabel('User number')
        ax.plot(X, Y)
        ax.legend()

    if channel_num_results:
        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in channel_num_results:
            step = STEP
            X.append(r[0]['channel_num'])
            Y.append(r[1]['final_system_cost'])
        ax.set_ylabel('System wide cost')
        ax.set_xlabel('Channel Number')
        ax.plot(X, Y)
        ax.legend()

        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in channel_num_results:
            step = STEP
            X.append(r[0]['channel_num'])
            Y.append(r[1]['final_beneficial_user_num'])
        ax.set_ylabel('Beneficial user')
        ax.set_xlabel('Channel Number')
        ax.plot(X, Y)
        ax.legend()

    plt.tight_layout()
    plt.show()
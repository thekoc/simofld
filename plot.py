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
STEP = 1
if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = max(Path('.').glob(r'results-*')).as_posix()
    with open(filename) as f:
        results = json.load(f)
    gamma_results = [r for r in results if r['group'] == 'gamma']
    lr_results = [r for r in results if r['group'] == 'lr']
    user_num_results = [r for r in results if r['group'] == 'user_num']
    channel_num_results = [r for r in results if r['group'] == 'channel_num']
    dq_results = [r for r in results if r['group'] == 'deep_q_learning']

    if gamma_results:
        fig, ax = plt.subplots()
        for r in gamma_results:
            step = STEP
            gamma = r['gamma']
            cost = r['result']['system_cost_histogram']
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')

            ax.plot(Y, label=f'gamma: {gamma:e}')
            # ax.set_ylim(bottom=0)
            ax.legend()
    
    if lr_results:
        fig, ax = plt.subplots()
        for r in lr_results:
            step = STEP
            lr = r['lr']
            cost = r['result']['system_cost_histogram']
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
            X.append(r['user_num'])
            Y.append(r['result']['final_system_cost'])
        ax.set_ylabel('System wide cost')
        ax.set_xlabel('User number')
        ax.plot(X, Y)
        ax.legend()

        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in user_num_results:
            step = STEP
            X.append(r['user_num'])
            Y.append(r['result']['final_beneficial_user_num'])
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
            X.append(r['channel_num'])
            Y.append(r['result']['final_system_cost'])
        ax.set_ylabel('System wide cost')
        ax.set_xlabel('Channel Number')
        ax.plot(X, Y)
        ax.legend()

        fig, ax = plt.subplots()
        X = []
        Y = []
        for r in channel_num_results:
            step = STEP
            X.append(r['channel_num'])
            Y.append(r['result']['final_beneficial_user_num'])
        ax.set_ylabel('Beneficial user')
        ax.set_xlabel('Channel Number')
        ax.plot(X, Y)
        ax.legend()


    if dq_results:
        fig, ax = plt.subplots()
        for r in dq_results:
            step = STEP
            cost = r['result']['system_cost_histogram']
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')
            ax.plot(Y, label=r.get('label', 'Deep Q Learning'))

            # ax.set_ylim(bottom=0)
            ax.legend()

    plt.tight_layout()
    plt.show()
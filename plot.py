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
        filenames = sys.argv[1:]
    else:
        filenames = [max(Path('.').glob(r'results-*')).as_posix()]
    results = []
    for filename in filenames:
        with open(filename) as f:
            results += json.load(f)
    
    convergence_results = [r for r in results if r['type'] == 'convergence']
    user_num_results = [r for r in results if r['type'] == 'user_num']
    channel_num_results = [r for r in results if r['type'] == 'channel_num']

    if convergence_results:
        for group in set(r['group'] for r in convergence_results):
            group_results = [r for r in convergence_results if r['group'] == group]
            fig, ax = plt.subplots()
            for r in group_results:
                step = STEP
                gamma =  r.get('gamma', 0)
                cost = r['result']['system_cost_histogram']
                Y = np.convolve(cost, np.ones(step)/step, mode='valid')
                ax.plot(Y, label=r.get('label', ''))
                # ax.set_ylim(bottom=0)
                ax.legend()


    if user_num_results:
        for group in set(r['group'] for r in user_num_results):
            print(group)
            group_results = [r for r in user_num_results if r['group'] == group]
            print(group_results)
            algorithms = set(r['algorithm'] for r in group_results)

            fig, ax = plt.subplots()
            ax.set_ylabel('System wide cost')
            ax.set_xlabel('User number')
            for a in algorithms:
                X = [r['user_num'] for r in group_results if r['algorithm'] == a]
                Y = [r['result']['final_system_cost'] for r in group_results if r['algorithm'] == a]
                ax.plot(X, Y, label=a)
            ax.legend()

            fig, ax = plt.subplots()
            ax.set_ylabel('Beneficial user')
            ax.set_xlabel('User number')
            for a in algorithms:
                X = [r['user_num'] for r in group_results if r['algorithm'] == a]
                Y = [r['result']['final_beneficial_user_num'] for r in group_results if r['algorithm'] == a]
                ax.plot(X, Y, label=a)
            ax.legend()

        if channel_num_results:
            for group in set(r['group'] for r in channel_num_results):
                group_results = [r for r in channel_num_results if r['group'] == group]
                algorithms = set(r['algorithm'] for r in group_results)

                fig, ax = plt.subplots()
                ax.set_ylabel('System wide cost')
                ax.set_xlabel('Channel number')
                for a in algorithms:
                    X = [r['channel_num'] for r in group_results if r['algorithm'] == a]
                    Y = [r['result']['final_system_cost'] for r in group_results if r['algorithm'] == a]
                    ax.plot(X, Y, label=a)
                ax.legend()

                fig, ax = plt.subplots()
                ax.set_ylabel('Beneficial user')
                ax.set_xlabel('Channel number')
                for a in algorithms:
                    X = [r['channel_num'] for r in group_results if r['algorithm'] == a]
                    Y = [r['result']['final_beneficial_user_num'] for r in group_results if r['algorithm'] == a]
                    ax.plot(X, Y, label=a)
                ax.legend()

    plt.tight_layout()
    plt.show()
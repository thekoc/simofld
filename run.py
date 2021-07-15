import sys
sys.path.insert(0, 'src')
import json
import numpy as np
from matplotlib import pyplot as plt

from simofld.masl import CloudServer, MobileUser, RayleighChannel


from simofld.envs import create_env
STEP = 1
if __name__ == '__main__':
    with open('results-20210715-035751.json') as f:
        results = json.load(f)
    gamma_results = [r for r in results if r[0]['group'] == 'gamma']
    lr_results = [r for r in results if r[0]['group'] == 'lr']
    fig, ax = plt.subplots()
    for r in gamma_results:
        step = STEP
        gamma = r[0]['gamma']
        cost = r[1]['system_cost_histogram']
        Y = np.convolve(cost, np.ones(step)/step, mode='valid')
        ax.plot(Y, label=f'gamma: {gamma:e}')
        ax.legend()
    
    fig, ax = plt.subplots()
    for r in lr_results:
        step = STEP
        lr = r[0]['lr']
        cost = r[1]['system_cost_histogram']
        Y = np.convolve(cost, np.ones(step)/step, mode='valid')
        ax.plot(Y, label=f'lr: {lr:e}')
        ax.legend()

    plt.tight_layout()
    plt.show()
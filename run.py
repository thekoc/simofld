import sys
sys.path.insert(0, 'src')
import json
import numpy as np
from matplotlib import pyplot as plt

from simofld.masl import CloudServer, MobileUser, RayleighChannel


from simofld.envs import create_env

if __name__ == '__main__':
    with open('results.json') as f:
        results = json.load(f)
    gamma_results = results[0:4]
    lr_results = results[4:8]
    fig, ax = plt.subplots()
    for r in gamma_results:
        step = 1
        gamma = r['gamma']
        cost = r['result']['system_cost_histogram']
        Y = np.convolve(cost, np.ones(step)/step, mode='valid')
        ax.plot(Y, label=f'gamma: {gamma:e}')
        ax.legend()
    
    fig, ax = plt.subplots()
    for r in lr_results:
        step = 1
        lr = r['lr']
        cost = r['result']['system_cost_histogram']
        Y = np.convolve(cost, np.ones(step)/step, mode='valid')
        ax.plot(Y, label=f'lr: {lr:e}')
        ax.legend()

    plt.tight_layout()
    plt.show()
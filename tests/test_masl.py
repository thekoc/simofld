
import sys
from pathlib import Path
src = Path(__file__).absolute().parent.parent.joinpath('src')
sys.path.insert(0, str(src))

import unittest
import json
from simofld.masl import CloudServer, MASLProfile, MobileUser, RayleighChannel, create_env
from simofld.masl import SIMULATION_PARAMETERS
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils
from numpy import random as np_random
import numpy as np
from matplotlib import pyplot as plt


UNTIL = 600

def test_paramater(gamma, lr, until, mobile_n=10, channel_n=5):
    np_random.seed(5)
    SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'] = gamma
    SIMULATION_PARAMETERS['LEARNING_RATE'] = lr
    channels = [RayleighChannel() for _ in range(channel_n)]
    nodes = [MobileUser(channels) for _ in range(mobile_n)]
    profile = MASLProfile(nodes, 1)
    step_interval = 1
    cloud_server = CloudServer()
    with create_env(nodes, cloud_server, profile, until=until, step_interval=step_interval) as env:
        env.run()
    
    for node in nodes:
        print(node._w.round(1) * 10)
        print(node._x)
    
    result = {
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': len([n for n in nodes if n._choice_index != 0])
    }
    return result

class TestMASL(unittest.TestCase):
    def test_main_algorithm_gamma(self):

        until = UNTIL
        result_list = []
        ## Gamma
        cost_list = []
        gammas = [10**4, 10**5, 10**6, 10**7]
        for gamma in gammas:
            lr = 0.1
            mobile_n = 20
            channel_n = 7
            result = test_paramater(gamma, lr, until=until, mobile_n=mobile_n, channel_n=channel_n)
            cost_list.append(result['system_cost_histogram'])
            result_list.append(dict(gamma=gamma, lr=lr, until=until, mobile_n=mobile_n, channel_n=channel_n, result=result))
        
        fig, ax = plt.subplots()

        for cost, gamma in zip(cost_list, gammas):
            # ax.plot(cost, label=f'1 gamma: {gamma:e}')
            step = 1
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')
            ax.plot(Y, label=f'gamma: {gamma:e}')
        ax.set_ylim(bottom=0)
        ax.legend(loc='best')
        plt.savefig('gamma.png')
        with open('results_gamma.json', 'w') as f:
            json.dump(result_list, f)

    def test_main_algorithm_lr(self):
        until = UNTIL
        ## Learning Rate
        result_list = []
        cost_list = []
        lr_list = [0.05, 0.1, 0.3, 0.5]
        for lr in lr_list:
            gamma = 10**5
            mobile_n = 20
            channel_n = 7
            result = test_paramater(gamma, lr, until=until, mobile_n=mobile_n, channel_n=channel_n)
            cost_list.append(result['system_cost_histogram'])
            result_list.append(dict(gamma=gamma, lr=lr, until=until, mobile_n=mobile_n, channel_n=channel_n, result=result))
        
        fig, ax = plt.subplots()

        for cost, lr in zip(cost_list, lr_list):
            step = 1
            Y = np.convolve(cost, np.ones(step)/step, mode='valid')
            ax.plot(Y, label=f'lr: {lr:e}')
        ax.set_ylim(bottom=0)
        ax.legend(loc='best')
        plt.savefig('lr.png')
        
        with open('results_lr.json', 'w') as f:
            json.dump(result_list, f)
        # plt.tight_layout()
        # plt.show()
        # fig, ax = plt.subplots()
        # for i in range(len(channels) + 1):
        #     ax.plot([w[i] for w in nodes[0]._w_history])
        # ax.legend(loc='best')
        # plt.show()
    
    def test_main_algorithm_user_channel_numbers(self):
        pass
        ## User number
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # results = []
        # for mobile_n in range(20, 45, 5):
        #     result = test_paramater(gamma=10**8, lr=0.1, until=600, mobile_n=mobile_n, channel_n=8)
        #     results.append(result)
        
        # ax1.plot(range(20, 45, 5), [r['final_system_cost'] for r in results])
        # ax1.set_ylabel('System cost')
        # ax1.set_xlabel('User number')
        
        # ax2.plot(range(20, 45, 5), [r['final_beneficial_user_num'] for r in results])
        # ax2.set_ylabel('Beneficial user number')
        # ax2.set_xlabel('User number')


        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # results = []
        # for channel_n in range(4, 15, 2):
        #     result = test_paramater(gamma=10**8, lr=0.1, until=600, mobile_n=40, channel_n=channel_n)
        #     results.append(result)
        
        # ax1.plot(range(4, 15, 2), [r['final_system_cost'] for r in results])
        # ax1.set_ylabel('System cost')
        # ax1.set_xlabel('Channel number')
        
        # ax2.plot(range(4, 15, 2), [r['final_beneficial_user_num'] for r in results])
        # ax2.set_ylabel('Beneficial user number')
        # ax2.set_xlabel('Channel number')


if __name__ == '__main__':
    unittest.main()
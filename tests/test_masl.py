
from numbers import Number
import sys
import time
from pathlib import Path
from typing import NamedTuple
src = Path(__file__).absolute().parent.parent.joinpath('src')
sys.path.insert(0, str(src))

import unittest
from multiprocessing import Pool, Queue
import json
from simofld.masl import CloudServer, MASLProfile, MobileUser, RayleighChannel, create_env
from simofld.masl import SIMULATION_PARAMETERS
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils
from numpy import random as np_random
import numpy as np
from matplotlib import pyplot as plt


UNTIL = 3

TestParameters = NamedTuple('TestParameters', [('group', str), ('gamma', Number), ('lr', Number), ('until', Number), ('user_num', int), ('channel_num', int)])


def test_paramater(parameters: TestParameters):
    np_random.seed(5)
    SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'] = parameters.gamma
    SIMULATION_PARAMETERS['LEARNING_RATE'] = parameters.lr
    channels = [RayleighChannel() for _ in range(parameters.channel_num)]
    nodes = [MobileUser(channels) for _ in range(parameters.user_num)]
    profile = MASLProfile(nodes, 1)
    step_interval = 1
    cloud_server = CloudServer()
    with create_env(nodes, cloud_server, profile, until=parameters.until, step_interval=step_interval) as env:
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


q = Queue(maxsize=1)
q.put(0)

def task(p):
    result = test_paramater(p)
    count = q.get()
    print('='*20 + '\n' + str(count + 1))
    q.put(count + 1)
    return result

class TestMASL(unittest.TestCase):
    def test_main_algorithm(self):
        until = UNTIL
        result_list = []
        user_num = 20
        channel_num = 7

        test_parameters_list = [
            TestParameters(group='gamma', gamma=5 * 10**4, lr=0.1, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='gamma', gamma=10**5, lr=0.1, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='gamma', gamma=5 * 10**5, lr=0.1, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='gamma', gamma=10**6, lr=0.1, until=until, user_num=user_num, channel_num=channel_num),

            TestParameters(group='lr', gamma=10**5, lr=0.06, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='lr', gamma=10**5, lr=0.1, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='lr', gamma=10**5, lr=0.3, until=until, user_num=user_num, channel_num=channel_num),
            TestParameters(group='lr', gamma=10**5, lr=0.5, until=until, user_num=user_num, channel_num=channel_num),
        ]



        # with Pool(2) as p:
        #     results = p.map(task, test_parameters_list)

        results = []
        for p in test_parameters_list[:1]:
            results += [task(p)]
        
        result_list = list(zip([p._asdict() for p in test_parameters_list], results))
        with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}.json', 'w') as f:
            json.dump(result_list, f)

        # for parameters in test_parameters_list:
        #     result = test_paramater(gamma=parameters.gamma, lr=parameters.lr, until=until, mobile_n=parameters.user_num, channel_n=parameters.channel_num)
        #     result_list.append(dict(parameters=parameters, result=result))
        #     # checkpoint
        #     with open('results.json', 'w') as f:
        #         json.dump(result_list, f)

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

print(__name__)

if __name__ == '__main__':
    TestMASL().test_main_algorithm()
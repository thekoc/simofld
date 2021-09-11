import sys
sys.path.insert(0, 'src')

import time
import json
import os
from numbers import Number
from multiprocessing import Pool


from numpy import random
import numpy as np
from simofld import masl, masl_deepq, br



def run_simulation(algorithm: str, user_num: int, channel_num: int, until: Number, profile_sample_interval: Number=10, distances=None, active_probabilities=None, simulation_parameters={}, **kwargs):
    if 'gamma' in kwargs:
        masl.SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'] = kwargs['gamma']
    if 'lr' in kwargs:
        masl.SIMULATION_PARAMETERS['LEARNING_RATE'] = kwargs['lr']
    
    algorithm_to_model = {
        'br': br,
        'masl': masl,
        'double_dqn': masl_deepq
    }
    
    model = algorithm_to_model[algorithm]
    
    channels = [model.RayleighChannel() for _ in range(channel_num)]
    if distances is None:
        distances = 5 + random.random(user_num) * 45
    if active_probabilities is None:
        active_probabilities = 1 - random.random(user_num)
        # active_probabilities = np.ones_like(active_probabilities)
    users = [model.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    if model is not br:
        cloud_server = model.CloudServer()
    else:
        cloud_server = model.CloudServer(users)

    profile = model.Profile(users, profile_sample_interval)
    with model.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    result = {
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': len([n for n in users if n._choice_index != 0]),
    }
    return result

def run_simulation_wrapper(kwargs):
    return run_simulation(**kwargs)

def run_simulation_repeat(repeat: int, parameter):
    with Pool(os.cpu_count()) as pool:
        results = pool.map(run_simulation_wrapper, [parameter] * repeat)
    
    repeat_result = {}
    for key in results[0]:
        array = np.array([result[key] for result in results])
        avg = array.mean(axis=0)
        if key == 'system_cost_histogram':
            avg = list(avg)
        repeat_result[key] = avg

    return {**parameter, 'result': repeat_result}
    

def test_different_gammas():
    repeat = 250
    parameters = [
        {'group': 'gamma', 'algorithm': 'masl', 'gamma': 1e3, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 10,},
        {'group': 'gamma', 'algorithm': 'masl', 'gamma': 1e4, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 10,},
        {'group': 'gamma', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 10,},
        {'group': 'gamma', 'algorithm': 'masl', 'gamma': 1e6, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 10,},
    ]
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-masl-gamma.json', 'w') as f:
        json.dump(results, f)


def test_different_user_numbers():
    repeat = 250
    parameters = [
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 20, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 25, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 35, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 40, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'user_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 45, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500},
    ]
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-masl-user-num.json', 'w') as f:
        json.dump(results, f)


def test_different_channel_numbers():
    repeat = 250
    parameters = [
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 4, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 6, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 8, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 10, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 12, 'until': 500, 'profile_sample_interval': 500},
        {'group': 'channel_num', 'algorithm': 'masl', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 14, 'until': 500, 'profile_sample_interval': 500},
    ]
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-masl-channel-num.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    test_different_channel_numbers()

import sys

sys.path.insert(0, 'src')

import time
import json
import os
from numbers import Number
from multiprocessing import Pool


from numpy import random
import numpy as np
import simofld

from simofld import masl, masl_deepq, br, random_selection, always_local



def run_simulation(algorithm: str, user_num: int, channel_num: int, until: Number, profile_sample_interval: Number=10, distances=None, active_probabilities=None, simulation_parameters={}, **kwargs):
    if 'gamma' in kwargs:
        masl.SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'] = kwargs['gamma']
    if 'lr' in kwargs:
        masl.SIMULATION_PARAMETERS['LEARNING_RATE'] = kwargs['lr']
    if 'profile_no_sample_until' in kwargs:
        profile_no_sample_until = kwargs['profile_no_sample_until']
    else:
        profile_no_sample_until = None

    if algorithm == 'dueling_double_dqn':
        masl.SIMULATION_PARAMETERS['ENABLE_DUELING'] = True
    elif algorithm == 'double_dqn':
        masl.SIMULATION_PARAMETERS['ENABLE_DUELING'] = False
    
    algorithm_to_model = {
        'br': br,
        'masl': masl,
        'double_dqn': masl_deepq,
        'dueling_double_dqn': masl_deepq,
        'random': random_selection,
        'local': always_local,
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

    profile = model.Profile(users, profile_sample_interval, no_sample_until=profile_no_sample_until)
    with model.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    
    final_beneficial_user_num = 0
    for i, u in enumerate(users):
        if u._choice_index != 0 and profile._node_costs[i][-1] < u.local_cost() * u.active_probability:
            final_beneficial_user_num += 1
    
    result = {
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': final_beneficial_user_num,
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


def test_different_parameters(parameters, suffix='', repeat=500):
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-{suffix}.json', 'w') as f:
        json.dump(results, f)


def test_different_user_numbers(algorithm, repeat=250):
    parameters = [
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 10, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 15, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 20, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 25, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'user_num', 'type': 'user_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': 35, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
    ]
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-{algorithm}-user-num.json', 'w') as f:
        json.dump(results, f)

def test_different_channel_numbers(algorithm, repeat=250):
    user_num = 20
    parameters = [
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 4, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 6, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 8, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 10, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 12, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
        {'group': 'channel_num', 'type': 'channel_num', 'algorithm': algorithm, 'gamma': 1e6, 'lr': 0.1, 'user_num': user_num, 'channel_num': 14, 'until': 500, 'profile_sample_interval': 500, 'profile_no_sample_until': 500},
    ]
    results = []
    for parameter in parameters:
        results.append(run_simulation_repeat(repeat, parameter))
        print(parameter)
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-{algorithm}-channel-num.json', 'w') as f:
        json.dump(results, f)


parameters_gamma = [
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e3, 'label': f'gamma: {1e3}', 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e4, 'label': f'gamma: {1e4}', 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e5, 'label': f'gamma: {1e5}', 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e6, 'label': f'gamma: {1e6}', 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,},
]

parameters_lr = [
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e5, 'label': f'lr: {0.05}', 'lr': 0.05, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e5, 'label': f'lr: {0.1}', 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e5, 'label': f'lr: {0.2}', 'lr': 0.2, 'user_num': 30, 'channel_num': 5, 'until': 500,},
    {'group': 'gamma', 'type': 'convergence', 'algorithm': 'masl', 'gamma': 1e5, 'label': f'lr: {0.3}', 'lr': 0.3, 'user_num': 30, 'channel_num': 5, 'until': 500,},
]

if __name__ == '__main__':
    repeat = 500
    test_different_parameters(parameters_gamma, suffix='gamma', repeat=repeat)
    test_different_parameters(parameters_lr, suffix='lr', repeat=repeat)
    test_different_channel_numbers('masl', repeat=repeat)
    test_different_user_numbers('masl', repeat=repeat)
    test_different_channel_numbers('br', repeat=repeat)
    test_different_user_numbers('br', repeat=repeat)
    test_different_channel_numbers('random', repeat=repeat)
    test_different_user_numbers('random', repeat=repeat)
    test_different_channel_numbers('local', repeat=repeat)
    test_different_user_numbers('local', repeat=repeat)
    test_different_parameters([{'group': 'dq', 'type': 'convergence', 'algorithm': 'double_dqn', 'label': f'double dqn', 'lr': 0.1, 'user_num': 20, 'channel_num': 5, 'until': 500,}], suffix='dq', repeat=8)
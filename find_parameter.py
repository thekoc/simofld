from numbers import Number
import sys
import time
from typing import List
sys.path.insert(0, 'src')

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool
import os
from numpy import random
import numpy as np
import json


from simofld.br import BRProfile
from simofld import masl
from simofld import br
from simofld import masl_deepq

def generate_profile(seed=None):
    random.seed(seed)
    model = masl
    until = 1
    step_interval = 1
    user_num = 30
    channel_num = 5
    # distances = 22.5 + random.random(user_num) * 5
    distances = 0 + random.random(user_num) * 50
    active_probabilities = 1 - random.random(user_num)
    channels = [model.RayleighChannel() for _ in range(channel_num)]
    users = [model.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = model.CloudServer()
    profile = model.MASLProfile(users, 2)
    with model.create_env(users, cloud_server, profile, until=until, step_interval=step_interval) as env:
        env.run()
    return {
        'users': users,
        'profile': profile,
    }

def main():
    x_ds = np.linspace(1, 20e8, 50)
    y_local_f = np.linspace(1, 20e8, 50)

    X, Y = np.meshgrid(x_ds, y_local_f)

    pairs = []
    Z = np.empty_like(X)
    for i in range(len(x_ds)):
        for j in range(len(y_local_f)):
            x = masl.SIMULATION_PARAMETERS['DATA_SIZE'] = X[j][i]
            y = Y[j][i]
            masl.SIMULATION_PARAMETERS['LOCAL_CPU_CAPABILITY'] = [y]
            result = generate_profile()
            users = result['users']
            profile: BRProfile = result['profile']
            local_cost = np.sum(u.local_cost() * u.active_probability for u in users)
            random_cost = profile._system_wide_cost_samples[0]
            diff = np.abs((local_cost/random_cost) - 1)
            if diff < 0.1:
                pairs.append([x, y, diff])
            Z[j][i] = diff
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')



    ax.set_xlabel('data size')
    ax.set_ylabel('frequency')
    print(pairs)

    fig, ax = plt.subplots()
    min_Y = np.argmin(Z, axis=0)
    ax.plot(x_ds, y_local_f[min_Y])
    plt.show()

    # for data_size in np.arange(1e8, 6e8, 1e8):
    #     masl.SIMULATION_PARAMETERS['DATA_SIZE'] = data_size
    #     result = generate_profile()
    #     users = result['users']
    #     profile: BRProfile = result['profile']
    #     plt.plot(profile._system_wide_cost_samples, label=f'datasize={data_size}', linewidth=1)
    #     # print(f'{data_size:e}')
    #     # print(, )

    # plt.legend()
    # plt.show()


def find_initial():
    total = 0
    for i in range(1000):
        profile: masl.MASLProfile = generate_profile()['profile']
        r = profile._system_wide_cost_samples[0]
        print(i, r)
        # if r > 10000:
        #     for i, user in enumerate(profile.nodes):
        #         print(f'* {user._x}, {profile._node_choices[i][0]}')
        #     break
        
        
        total += r
    
    print(total / i)

def w_history():
    random.seed(1)
    model = masl
    until = 500
    step_interval = 1
    user_num = 30
    channel_num = 5
    # distances = 22.5 + random.random(user_num) * 5
    distances = 1 + random.random(user_num) * 49
    active_probabilities = 1 - random.random(user_num)
    channels = [model.RayleighChannel() for _ in range(channel_num)]
    users = [model.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = model.CloudServer()
    profile = model.MASLProfile(users, 1)
    with model.create_env(users, cloud_server, profile, until=until, step_interval=step_interval) as env:
        env.run()
    for j in range(3):
        plt.figure()
        w_history = np.array(users[j]._w_history)
        for i in range(w_history.shape[1]):
            plt.plot(w_history[:, i][::10], label=f'channel: {i}', marker=i)
    plt.legend()
    plt.show()

def run_br(group: str, user_num: int, channel_num: int, until: Number, profile_sample_interval: Number=10, **kwargs):
    channels = [br.RayleighChannel() for _ in range(channel_num)]
    # distances = 25 + random.random(user_num) * 0
    distances = 5 + random.random(user_num) * 45
    active_probabilities = 1 - random.random(user_num)
    # active_probabilities = np.ones_like(active_probabilities)
    users = [br.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = br.CloudServer(users)
    profile = br.BRProfile(users, profile_sample_interval)

    with br.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    result = {
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': len([n for n in users if n._choice_index != 0])
    }
    return result

def run_br_wrapper(p_dict: dict):
    return run_br(**p_dict)

def test_br():
    parameters = [
        {
            'group': 'gamma', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 500,
        }
    ]

    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(run_br_wrapper, parameters * 2)
    
    samples_na = np.array([result['system_cost_histogram'] for result in results])
    system_cost_histogram = list(samples_na.mean(axis=0))
    with open(f'results-test-br-{time.strftime("%Y%m%d-%H%M%S")}.json', 'w') as f:
        json.dump([{**parameters[0], 'result': {'system_cost_histogram': system_cost_histogram}}], f)

def run_masl(group: str, gamma: Number, lr: Number, user_num: int, channel_num: int, until: Number, profile_sample_interval: Number=10):
    masl.SIMULATION_PARAMETERS['LEARNING_RATE'] = lr
    masl.SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'] = gamma
    channels = [br.RayleighChannel() for _ in range(channel_num)]
    # distances = 25 + random.random(user_num) * 0
    distances = 5 + random.random(user_num) * 45
    active_probabilities = 1 - random.random(user_num)
    # active_probabilities = np.ones_like(active_probabilities)
    users = [masl.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = masl.CloudServer()
    profile = masl.MASLProfile(users, profile_sample_interval)

    with masl.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    result = {
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': len([n for n in users if n._choice_index != 0]),
        'profile': profile
    }
    return result

def run_masl_wrapper(p_dict: dict):
    return run_masl(**p_dict)

def test_masl():
    parameters = [
        {
            'group': 'gamma', 'gamma': 1e5, 'lr': 0.03, 'user_num': 30, 'channel_num': 5, 'until': 700, 'profile_sample_interval': 20
        }
    ]

    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(run_masl_wrapper, parameters * 10)
    
    samples_na = np.array([result['system_cost_histogram'] for result in results])
    system_cost_histogram = list(samples_na.mean(axis=0))
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}.json', 'w') as f:
        json.dump([{**parameters[0], 'result': {'system_cost_histogram': system_cost_histogram}}], f)

def test_cost_func():
    user_num = 30
    channel_num = 5
    profile_sample_interval = 10
    until = 1

    static_choice = masl.MASLProfile.system_wide_cost_vectorized_static_choice
    dynamic_choice = masl.MASLProfile.system_wide_cost_vectorized

    channels = [br.RayleighChannel() for _ in range(channel_num)]
    distances = 5 + random.random(user_num) * 45
    active_probabilities = 1 - random.random(user_num)
    users = [masl.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
    cloud_server = masl.CloudServer()

    for user in users:
        i = random.randint(0, len(channels) + 1)
        user._w = np.zeros(len(channels) + 1)
        user._w[0] = 1

    profile = masl.MASLProfile(users, profile_sample_interval)
    masl.MASLProfile.system_wide_cost = static_choice
    with masl.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    print(profile._system_wide_cost_samples[0])

    profile = masl.MASLProfile(users, profile_sample_interval)
    masl.MASLProfile.system_wide_cost = dynamic_choice
    with masl.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    print(profile._system_wide_cost_samples[0])

def test_local_cost():
    p = {
        'group': 'gamma', 'gamma': 1e5, 'lr': 0.1, 'user_num': 30, 'channel_num': 5, 'until': 1500, 'profile_sample_interval': 1499
    }

    result = run_masl(**p)
    profile: masl.MASLProfile = result['profile']
    users = profile.nodes
    user_costs = profile._node_costs
    for i, u in enumerate(users):
        print(f'cost of user {i}: {user_costs[i][-1]:.2f}, local: {u.local_cost() * u.active_probability:.2f}')
    
    print(f'total: {profile._system_wide_cost_samples[-1]}, local: {sum([u.local_cost() * u.active_probability for u in users])}')

def test_epsilon():
    pass


def run_dq(group: str, user_num: int, channel_num: int, until: Number, profile_sample_interval: Number=10, distances=None, active_probabilities=None, run_until_times=None, **kwargs):
    channels = [br.RayleighChannel() for _ in range(channel_num)]
    if distances is None:
        distances = 5 + random.random(user_num) * 45
    if active_probabilities is None:
        active_probabilities = 1 - random.random(user_num)
        # active_probabilities = np.ones_like(active_probabilities)
    users = [masl_deepq.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]

    if run_until_times:
        for i, run_until in enumerate(run_until_times):
            users[i]._run_until = run_until
    
    cloud_server = masl_deepq.CloudServer()
    profile = masl_deepq.Profile(users, profile_sample_interval)

    with masl_deepq.create_env(users, cloud_server, profile, until, 1) as env:
        env.run()
    result = {
        'group': group,
        'system_cost_histogram': profile._system_wide_cost_samples,
        'final_system_cost': profile._system_wide_cost_samples[-1],
        'final_beneficial_user_num': len([n for n in users if n._choice_index != 0]),
    }
    return result

def run_dq_wrapper(p_dict: dict):
    return run_dq(**p_dict)

def test_dq():
    parameters = [
        {
            'group': 'deep_q_learning', 'user_num': 30, 'channel_num': 5, 'until': 500, 'profile_sample_interval': 1
        }
    ]
    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(run_dq_wrapper, parameters * 1)
    # results = [run_dq_wrapper(parameters[0])]
    samples_na = np.array([result['system_cost_histogram'] for result in results])
    system_cost_histogram = list(samples_na.mean(axis=0))
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-dq.json', 'w') as f:
        json.dump([{**parameters[0], 'result': {'system_cost_histogram': system_cost_histogram}}], f)

def test_adaptiveness():
    parameters = [
    ]

    repeat = 5
    
    user_num = 2
    remain_user_num = 1

    channel_num = 1
    
    distances_array = 5 + random.random((repeat, user_num)) * 45
    active_probabilities_array = 1 - random.random((repeat, user_num))

    until = 30
    stop_time = 10


    for i in range(repeat):
        parameters.append({
            'group': 'deep_q_learning', 'label': 'Directly run',
            'user_num': remain_user_num, 'channel_num': channel_num, 'until': until,'profile_sample_interval': 2,
            'distances': list(distances_array[i][:remain_user_num]), 'activity_probabilities': list(active_probabilities_array[i][:remain_user_num]),
        })

    for i in range(repeat):
        run_until_times = [None] * remain_user_num + [stop_time] * (user_num - remain_user_num)
        parameters.append({
            'group': 'deep_q_learning', 'label': 'After manually stop',
            'user_num': user_num, 'channel_num': channel_num, 'until': until,'profile_sample_interval': 2,
            'distances': list(distances_array[i]), 'activity_probabilities': list(active_probabilities_array[i]),
            'run_until_times': run_until_times
        })
    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(run_dq_wrapper, parameters)
    
    json_results = []
    samples_na = np.array([result['system_cost_histogram'] for result in results[:repeat]])
    system_cost_histogram = list(samples_na.mean(axis=0))
    json_results.append(
        {**parameters[0], 'result': {'system_cost_histogram': system_cost_histogram}}
    )

    samples_na = np.array([result['system_cost_histogram'] for result in results[repeat:]])
    system_cost_histogram = list(samples_na.mean(axis=0))
    json_results.append(
        {**parameters[repeat], 'result': {'system_cost_histogram': system_cost_histogram}}
    )
    with open(f'results-{time.strftime("%Y%m%d-%H%M%S")}-dq.json', 'w') as f:
        json.dump(json_results, f)


if __name__ == '__main__':
    test_adaptiveness()

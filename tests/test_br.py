
from numbers import Number
import sys
import time
from pathlib import Path
from typing import NamedTuple
src = Path(__file__).absolute().parent.parent.joinpath('src')
sys.path.insert(0, str(src))

import unittest
from multiprocessing import Pool, Queue
from simofld.br import BRProfile, CloudServer, MobileUser, RayleighChannel, create_env as create_br_env
from simofld import masl
from simofld import envs, exceptions
from simofld.envs import create_env
import simofld.utils as utils
from numpy import random as np_random
import numpy as np
from matplotlib import pyplot as plt

class TestBR(unittest.TestCase):
    def test_will_update(self):
        users = [MobileUser([], 1) for _ in range(3)]
        cloud_server = CloudServer(users)
        results = {}
        async def f(i, delay, rtu):
            await envs.sleep(delay)
            value = (await cloud_server.will_update(rtu)).value
            results[i] = value
        
        with create_env([f(1, 1, True), f(2, 1, True), f(3, 1, False)], 0, 10) as env:
            env.run()
    
    def test_main_algorithm(self):
        np_random.seed(0)
        user_num = 30
        channels = [RayleighChannel() for _ in range(5)]
        # distances = 25 + np_random.random(user_num) * 0
        distances = 10 + np_random.random(user_num) * 40

        active_probabilities = 1 - np_random.random(user_num)
        # active_probabilities = np.ones_like(active_probabilities)
        users = [MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
        cloud_server = CloudServer(users)
        profile = BRProfile(users, 1)

        with create_br_env(users, cloud_server, profile, 500, 1) as env:
            env.run()
        plt.plot(profile._system_wide_cost_samples, label='br')
        # for i, choices in enumerate(profile._node_choices):
        #     plt.plot(np.array(choices) + 0.01 * i, label=i, linewidth=0.4)

        channels = [RayleighChannel() for _ in range(5)]
        users = [masl.MobileUser(channels, distance, active_probability) for distance, active_probability in zip(distances, active_probabilities)]
        cloud_server = masl.CloudServer()
        profile = masl.MASLProfile(users, 1)

        with masl.create_env(users, cloud_server, profile, 300, 1) as env:
            env.run()

        plt.plot(profile._system_wide_cost_samples, label='masl')
        for u in users:
            print(u._w.round(1))



        plt.legend()
        plt.tight_layout()
        plt.show()
        

if __name__ == '__main__':
    TestBR().test_main_algorithm()
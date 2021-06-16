"""This module implements MASL-algorithm from:
    J. Zheng, Y. Cai, Y. Wu and X. Shen, "Dynamic Computation Offloading for Mobile Cloud Computing: A Stochastic Game-Theoretic Approach," in IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 771-786, 1 April 2019, doi: 10.1109/TMC.2018.2847337.
"""
from typing import List, Optional
from numbers import Number

import numpy as np
from numpy import random

from . import envs
from .model import Node, Channel

SIMULATION_PARAMETERS = {
    # CHART
    'AP_COVERAGE': 50, # meter
    'MOBILE_NUM': 20,
    'MOBILE_ACTIVE_PROBABILITY': 0.9,
    'CHANNEL_NUM': 10,
    'CHANNEL_BANDWITH': 5, # MHz
    'TRANSMIT_POWER': 100, # mW
    'PATH_LOSS_EXPONENT': 4,
    'BACKGROUND_NOISE': -100, # dBm
    'DATA_SIZE': 5000, # KB
    'LOCAL_CPU_CYCLES': 1000, # Megacycles
    'CLOUD_CPU_CYCLES': 1200, # Megacycles
    'COMPUTATIONAL_ENERGY_WEIGHT': (0, 0.5, 1.0),
    'COMPUTATIONAL_TIME_WEIGHT': None,
    'COMPUTING_ENERGY_EFFECIENCY': (400, 500, 600), # Megacycles/J
    
    # MOBILE
    'LEARNING_RATE': 0.1,
    
    # Channel
    'CHANNEL_SCALING_FACTOR': 10**5,
}

def I():
    return 1

def Q():
    return 1

class MobileUser(Node):
    def __init__(self, data_process_rate: Optional[Number], channels: List[Channel]) -> None:
        self._x = random.random() * 50
        self.lr = SIMULATION_PARAMETERS['LEARNING_RATE']
        self.active_probability = 1 - random.random()
        self.channels = channels
        self.active = None
        super().__init__(data_process_rate=data_process_rate)

    async def perform_cloud_computation(self, cloud_server: 'CloudServer', channel: Channel, upload_duration: Number):
        env = self.get_current_env()
        data = await channel.transfer_data(self, cloud_server, duration=upload_duration)
        env.create_task(cloud_server.compute(datasize=data.size))

    async def perform_local_computation(self, duration: Number):
        await self.compute(duration=duration)

    async def main_loop(self):
        channel_num = len(self.channels)
        step_lapse = 1
        lr = self.lr
        choice_list = [None] + self.channels
        
        cloud_server = self.get_current_env().g.cloud_server

        w = np.full(channel_num + 1, 1 / (channel_num + 1))
        theta = self.active_probability
        gamma = 10**5

        while True:
            self.active = True if random.random() < theta else False
            if self.active:
                choice_index = random.choice(channel_num + 1, 1, p=w).item()
                if choice_index == 0: # local execution
                    payoff = I()
                    await self.perform_local_computation(step_lapse)
                    
                else: # cloud execution
                    channel = choice_list[choice_index]
                    await self.perform_cloud_computation(cloud_server, channel, step_lapse)
                    payoff = Q()

                e = np.zeros(channel_num + 1)
                e[choice_index] = 1

                r = 1 - gamma * payoff
                w = w + lr * r * (e - w)
            else:
                await envs.sleep(step_lapse)


class CloudServer(Node):
    def __init__(self, data_process_rate: Number = 1) -> None:
        super().__init__(data_process_rate=data_process_rate)
        env =  self.get_current_env()
        if env.g.cloud_server is not None:
            raise ValueError('Only one cloud server can be set for the env')
        env.g.cloud_server = self

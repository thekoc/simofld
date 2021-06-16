"""This module implements MASL-algorithm from:
    J. Zheng, Y. Cai, Y. Wu and X. Shen, "Dynamic Computation Offloading for Mobile Cloud Computing: A Stochastic Game-Theoretic Approach," in IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 771-786, 1 April 2019, doi: 10.1109/TMC.2018.2847337.
"""
from typing import List, Optional
from numbers import Number

import numpy as np
from numpy import random

from . import envs
from .model import Node, Channel

class MobileUser(Node):
    def __init__(self, data_process_rate: Optional[Number], channels: List[Channel]) -> None:
        self.channels = channels
        self.active = True
        super().__init__(data_process_rate=data_process_rate)
    
    async def perform_cloud_computation(self, cloud_server: 'CloudServer', channel: Channel, upload_duration: Number):
        env = self.current_env()
        data = await channel.transfer_data(self, cloud_server, duration=upload_duration)
        env.create_task(cloud_server.compute(data.size))

    async def perform_local_computation(self, compute_duration: Number):
        env = self.current_env()
        await envs.sleep(compute_duration)

    async def main_loop(self):
        channel_num = len(self.channels)
        step_lapse = 1
        lr = 0.1 # learning rate
        choice_list = [None] + self.channels


        w = np.full(channel_num + 1, 1 / (channel_num + 1))

        while True:
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
    pass
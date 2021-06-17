"""This module implements MASL-algorithm from:
    J. Zheng, Y. Cai, Y. Wu and X. Shen, "Dynamic Computation Offloading for Mobile Cloud Computing: A Stochastic Game-Theoretic Approach," in IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 771-786, 1 April 2019, doi: 10.1109/TMC.2018.2847337.
"""
from typing import List, Optional
from numbers import Number

import numpy as np
from numpy import log2, random

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

class MobileUser(Node):
    def __init__(self, channels: List[Channel]) -> None:
        super().__init__()

        self.data_process_rate = 
        self._x = random.random() * 50
        self.lr = SIMULATION_PARAMETERS['LEARNING_RATE']
        self.active_probability = 1 - random.random()
        self.channels = channels
        self.transmit_power = SIMULATION_PARAMETERS['SIMULATION_PARAMETERS'] # p_i
        self.active = None

        # TODO: Make them stochastic
        self._payoff_weight_energy = 0.5
        self._payoff_weight_time = 0.5

        self._datasize = SIMULATION_PARAMETERS['DATA_SIZE']




    
    def _psi(self, bandwidth: Number) -> Number:
        """Used to calculate `self._I`.

        Args:
            bandwidth (Number): Bandwidth of chosen channel

        Returns:
            Number: :math:`\psi`
        """
        mu_E = self._payoff_weight_energy
        mu_T = self._payoff_weight_time
        p = self.transmit_power
        C = self._datasize
        B = bandwidth
        

        T_loc = C / self.data_process_rate
        E_loc = T_loc 

        T_clo_2 = C / self.data_process_rate

        n = (mu_E * p + mu_T) * C
        d = B * (mu_T * T_loc + mu_E * E_loc - mu_T * T_clo_2)
        return n / d

    def _Q(self, channel: 'RayleighChannel') -> Number:
        channel_power = channel.channel_power(self)
        bandwidth = channel.bandwidth
        psi = self._psi(bandwidth)
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']
        return (channel_power / (2**psi - 1)) - sigma_0

    def _I(self, channel: 'RayleighChannel') -> Number:
        return channel.total_channel_power() - channel.channel_power(self)

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
            
            # Use this function to stay sync with other nodes
            await envs.wait_for_simul_tasks()

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


class RayleighChannel(Channel):
    """A channel that follows Rayleigh fading. Notice that the datarate will also be affected by connected user. 
    """
    def __init__(self) -> None:
        super().__init__()
        # TODO: Update it periodically
        self.beta: Number = random.exponential(1) # Rayleigh fading factor
        self.bandwidth: Number = SIMULATION_PARAMETERS['CHANNEL_BANDWITH']

    def channel_power(self, mobile: MobileUser) -> Number:
        """It is the :math:`p_i g_{i,o}` in the original paper.

        Args:
            mobile (MobileUser): Mobile user to be calculated on.
        Returns:
            Number: :math:`p_i g_{i,o}`
        """
        beta = self.beta
        alpha = SIMULATION_PARAMETERS['PATH_LOSS_EXPONENT']
        distance = abs(mobile._x)
        return mobile.transmit_power * distance**(-alpha) * beta
    
    def total_channel_power(self) -> Number:
        """Total channel_power for active user that are using this channel. :math:`\sum_{i \in \mathcal{A}} p_i g_{i,o}`

        Returns:
            Number: :math:`\sum_{i \in \mathcal{A}} p_i g_{i,o}`
        """
        # TODO: May take irrelevant into account?
        beta = self.beta
        tot_power = 0
        for transmission in self.transmission_list:
            node = transmission.from_node
            assert isinstance(node, MobileUser)
            if node.active:
                tot_power += self.channel_power(node, beta)
        return tot_power


    def datarate_between(self, mobile: MobileUser, to_node: 'CloudServer') -> Number:
        beta = self.beta
        channel_power = self.channel_power(mobile, beta)

        total_channel_power = self.total_channel_power(beta)
        
        B = self.bandwidth
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']

        return B * log2(1 + channel_power / (total_channel_power - channel_power + sigma_0))
        
    


class CloudServer(Node):
    def __init__(self, data_process_rate: Number = 1) -> None:
        super().__init__(data_process_rate=data_process_rate)


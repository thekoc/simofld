"""This module implements MASL-algorithm from:
    J. Zheng, Y. Cai, Y. Wu and X. Shen, "Dynamic Computation Offloading for Mobile Cloud Computing: A Stochastic Game-Theoretic Approach," in IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 771-786, 1 April 2019, doi: 10.1109/TMC.2018.2847337.
"""
import logging
from typing import List, Optional
from numbers import Number
from logging import getLogger

import numpy as np
from numpy import log2, random

from . import envs
from .model import LocalData, Node, Channel

SIMULATION_PARAMETERS = {
    # CHART
    'AP_COVERAGE': 20, # meter
    'MOBILE_NUM': 20,
    'MOBILE_ACTIVE_PROBABILITY': 0.9,
    'CHANNEL_NUM': 10,
    'CHANNEL_BANDWITH': 5 * 10**6, # MHz
    'TRANSMIT_POWER': 100 * 10**-3, # mW
    'PATH_LOSS_EXPONENT': 4,
    'BACKGROUND_NOISE': 10**-13, # dBm
    'DATA_SIZE': 5000 * 10**3, # KB
    'LOCAL_CPU_CYCLES': 1000 * 10**6, # Megacycles
    'CLOUD_CPU_CYCLES': 1200 * 10**6, # Megacycles
    'LOCAL_CPU_CAPABILITY': (0.5 * 10**9, 0.8 * 10**9, 1.0 * 10**9), # GHz,
    'CLOUD_CPU_CAPABILITY': 12 * 10**9, # GHz,
    'COMPUTATIONAL_ENERGY_WEIGHT': (0, 0.5, 1.0),
    'COMPUTATIONAL_TIME_WEIGHT': None,
    'COMPUTING_ENERGY_EFFECIENCY': (400 * 10**6, 500 * 10**6, 600 * 10**6), # Megacycles/J
    
    # MOBILE
    'LEARNING_RATE': 0.03,
    
    # Channel
    'CHANNEL_SCALING_FACTOR': 10000,
}

logger = getLogger(__name__)

class MobileUser(Node):
    def __init__(self, channels: List[Channel]) -> None:
        super().__init__()
        self._x = (1 - random.random()) * SIMULATION_PARAMETERS['AP_COVERAGE'] # Distance to the AP
        self._w = None
        self._w_history = []
        self.lr = SIMULATION_PARAMETERS['LEARNING_RATE']
        self.active_probability = 1 - random.random()
        self.active_probability = 1
        self.channels = channels
        self.transmit_power = SIMULATION_PARAMETERS['TRANSMIT_POWER'] # p_i
        self.active = None

        self.cpu_frequency = SIMULATION_PARAMETERS['LOCAL_CPU_CAPABILITY'][0] # Megacycles/s
        self.cpu_effeciency = SIMULATION_PARAMETERS['COMPUTING_ENERGY_EFFECIENCY'][0] # Megacycles/J TODO: Random selection
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

        D_loc = SIMULATION_PARAMETERS['LOCAL_CPU_CYCLES']
        T_loc = D_loc / (self.cpu_frequency)
        E_loc = D_loc / self.cpu_effeciency 

        D_clo = SIMULATION_PARAMETERS['CLOUD_CPU_CYCLES']
        T_clo_2 = D_clo / (SIMULATION_PARAMETERS['CLOUD_CPU_CAPABILITY'])

        n = (mu_E * p + mu_T) * C
        d = B * (mu_T * T_loc + mu_E * E_loc - mu_T * T_clo_2)
        return n / d

    def _Q(self) -> Number:
        channel_power = RayleighChannel.channel_power(self)
        bandwidth = RayleighChannel.bandwidth
        psi = self._psi(bandwidth)
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']
        return (channel_power / (2**psi - 1)) - sigma_0

    def _I(self, channel: 'RayleighChannel') -> Number:
        return channel.total_channel_power(exclude=self)

    async def perform_cloud_computation(self, cloud_server: 'CloudServer', channel: Channel, upload_duration: Number):
        env = self.get_current_env()
        data = await channel.transfer_data(self, cloud_server, duration=upload_duration)
        transmission_datasize = data.size

        # TODO: Make this more readable
        total_size = SIMULATION_PARAMETERS['DATA_SIZE']
        total_cycles = SIMULATION_PARAMETERS['CLOUD_CPU_CYCLES']
        cloud_frequency = cloud_server.cpu_frequency
        
        task_cycle_cloud =  transmission_datasize / total_size * total_cycles
        duration = task_cycle_cloud / cloud_frequency
        env.create_task(cloud_server.compute(duration))

    async def perform_local_computation(self, duration: Number):
        await self.compute(duration=duration)

    async def main_loop(self):
        channel_num = len(self.channels)
        step_lapse = 0.01
        lr = self.lr
        choice_list = [None] + self.channels
        
        cloud_server: CloudServer = self.get_current_env().g.cloud_server

        self._w = np.full(channel_num + 1, 1 / (channel_num + 1))
        w = self._w
        theta = self.active_probability
        gamma = SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR']

        while True:
            self.active = True if random.random() < theta else False

            if self.active:
                choice_index = random.choice(channel_num + 1, 1, p=w).item()

                if choice_index == 0: # local execution
                    logger.debug(f'Calculating Q for user {self.id}')
                    ongoing_trans = None
                    payoff = self._Q()
                    await self.perform_local_computation(step_lapse)
                    logger.debug(f'Q: {payoff}')
                    logger.debug(f'I: {self._I(self.channels[0])}')

                    
                else: # cloud execution
                    channel = choice_list[choice_index]
                    ongoing_trans = [t.from_node.id for t in channel.transmission_list]

                    payoff = self._I(channel)
                    logger.debug(f'Calculating I for user {self.id}, result:{payoff},  channel: {channel.id} list: {ongoing_trans}')

                    await self.perform_cloud_computation(cloud_server, channel, step_lapse)
                    logger.debug(f'I: {payoff}')

                e = np.zeros(channel_num + 1)
                e[choice_index] = 1

                logger.debug('='*20)
                logger.debug(f'User {self.id}, x: {self._x} chooses channel {choice_index}, now: {envs.get_current_env().now}')
                logger.debug(f'onging trans {ongoing_trans}')

                logger.debug(w)
                r = 1 - gamma * payoff
                logger.debug(f'index: {choice_index}, payoff: {payoff}, r: {r}')
                logger.debug(f'time: {cloud_server.total_compute_time}')
                logger.debug('='*20 + '\n'*2)
                w = w + lr * r * (e - w)
                self._w = w
                self._w_history.append(w)
                assert r > 0
            else:
                await envs.sleep(step_lapse)


class RayleighChannel(Channel):
    """A channel that follows Rayleigh fading. Notice that the datarate will also be affected by connected user. 
    """
    bandwidth: Number = SIMULATION_PARAMETERS['CHANNEL_BANDWITH']

    # TODO: Update it periodically
    beta: Number = random.exponential(1) # Rayleigh fading factor

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def channel_power(cls, mobile: MobileUser) -> Number:
        """It is the :math:`p_i g_{i,o}` in the original paper.

        Args:
            mobile (MobileUser): Mobile user to be calculated on.
        Returns:
            Number: :math:`p_i g_{i,o}`
        """
        beta = cls.beta
        alpha = SIMULATION_PARAMETERS['PATH_LOSS_EXPONENT']
        distance = abs(mobile._x)
        result = mobile.transmit_power * distance**(-alpha) * beta
        logger.debug(f'chnnel power {result}, beta {beta}')
        return result
    
    def total_channel_power(self, exclude: Optional[MobileUser] = None) -> Number:
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
            if node.active and (exclude is not node):
                tot_power += self.channel_power(node)
        return tot_power


    def datarate_between(self, mobile: MobileUser, to_node: 'CloudServer') -> Number:
        channel_power = self.channel_power(mobile)

        total_channel_power = self.total_channel_power(exclude=mobile)
        B = self.bandwidth
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']

        result = B * log2(1 + channel_power / (total_channel_power + sigma_0))
        return result

class CloudServer(Node):
    def __init__(self) -> None:
        super().__init__()
        self.cpu_frequency: Number = SIMULATION_PARAMETERS['CLOUD_CPU_CAPABILITY'] # Megacycles/s
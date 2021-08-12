"""This module implements MASL-algorithm from:
    J. Zheng, Y. Cai, Y. Wu and X. Shen, "Dynamic Computation Offloading for Mobile Cloud Computing: A Stochastic Game-Theoretic Approach," in IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 771-786, 1 April 2019, doi: 10.1109/TMC.2018.2847337.
"""
import logging
import time
from typing import List, Optional
from functools import lru_cache
from numbers import Number
from logging import getLogger

import numpy as np
from numpy import log2, random

from . import envs
from .model import Node, Channel, Profile, SimulationEnvironment, Transmission

SIMULATION_PARAMETERS = {
    # CHART
    'AP_COVERAGE': 40, # meter

    'CHANNEL_NUM': 10,
    'CHANNEL_BANDWITH': 5e6, # MHz
    'TRANSMIT_POWER': 100e-3, # mW
    'PATH_LOSS_EXPONENT': 4,
    'BACKGROUND_NOISE': 1e-13 , # dBm
    'DATA_SIZE': 8*5000e3, # KB
    'LOCAL_CPU_CYCLES': 1000e6, # Megacycles
    'CLOUD_CPU_CYCLES': 1200e6, # Megacycles
    'LOCAL_CPU_CAPABILITY': (0.5e9, 0.8e9, 1.0e9), # GHz,
    'CLOUD_CPU_CAPABILITY': 12e9, # GHz,
    'COMPUTATIONAL_ENERGY_WEIGHT': (0, 0.5, 1.0),
    'COMPUTATIONAL_TIME_WEIGHT': None,
    'COMPUTING_ENERGY_EFFECIENCY': (400e6, 500e6, 600e6), # Megacycles/J
    
    # MOBILE
    'LEARNING_RATE': 0.1,
    
    # Channel
    'CHANNEL_SCALING_FACTOR': 2e6,
}

logger = getLogger(__name__)

approx_betas = np.linspace(0.5, 5, 20)
def _generate_exponential(size: Number):
    def _map(v: Number):
        if v > 5:
            return v
        for n in approx_betas:
            if v <= n:
                return n
    if size == 1:
        return _map(random.standard_exponential(1))
    else:
        return [_map(v) for v in random.standard_exponential(size)]

@lru_cache(maxsize=999)
def _generate_rayleigh_factor(mobile, getnow):
    beta = _generate_exponential(1)
    return beta

class MobileUser(Node):
    def __init__(self, channels: List[Channel], distance=None, active_probability=None) -> None:
        super().__init__()
        # distance to AP
        self._x = 10 + (1 - random.random()) * SIMULATION_PARAMETERS['AP_COVERAGE'] if distance is None else distance
        # self._x = 5

        # Probability for chosing different channels (0 means local computation).
        # Will be updated every iteration
        self._w = None
        self._w_history = []

        self._choice_index = None
        self.lr = SIMULATION_PARAMETERS['LEARNING_RATE']
        self.active_probability = 1 - random.random() if active_probability is None else active_probability
        # self.active_probability = 1
        self.channels = channels
        self.transmit_power = SIMULATION_PARAMETERS['TRANSMIT_POWER'] # p_i
        self.active = None

        self.cpu_frequency = SIMULATION_PARAMETERS['LOCAL_CPU_CAPABILITY'][0] # Megacycles/s
        self.cpu_effeciency = SIMULATION_PARAMETERS['COMPUTING_ENERGY_EFFECIENCY'][0] # Megacycles/J TODO: Random selection
        
        # The weights to calculate payoff function
        self._payoff_weight_energy = 0
        self._payoff_weight_time = 1 - self._payoff_weight_energy

        self._datasize = SIMULATION_PARAMETERS['DATA_SIZE']

    def _psi(self, bandwidth: Number) -> Number:
        """Used to calculate `self._I()`.

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
        result = (channel_power / (2**psi - 1)) - sigma_0
        return result

    def _I(self, channel: 'RayleighChannel') -> Number:
        return channel.total_channel_power(exclude=self)
    
    def _get_cloud_server(self) -> 'CloudServer':
        """Get the cloud server for the system. (Only one cloud server for this model)

        Returns:
            CloudServer: Cloudserver
        """
        cloud_server: CloudServer = self.get_current_env().g.cloud_server
        return cloud_server

    def cloud_cost(self, datarate: Number) -> Number:
        mu_E = self._payoff_weight_energy
        mu_T = self._payoff_weight_time
        C = self._datasize
        cloud_server: 'CloudServer' = self.get_current_env().g.cloud_server

        D_clo = SIMULATION_PARAMETERS['CLOUD_CPU_CYCLES']
        F_clo = cloud_server.cpu_frequency
        cloud_duration = D_clo / F_clo

        transmission_duration = C / datarate

        total_duration = transmission_duration + cloud_duration
        total_energy = self.transmit_power * transmission_duration

        return mu_T * total_duration + mu_E * total_energy
    
    def local_cost(self) -> Number:
        """Local cost. It is a constant number for the original paper.

        Returns:
            Number: Local cost number.
        """
        mu_E = self._payoff_weight_energy
        mu_T = self._payoff_weight_time
        D_loc = SIMULATION_PARAMETERS['LOCAL_CPU_CYCLES']
        F_loc = self.cpu_frequency
        T_loc = D_loc / F_loc
        E_loc = D_loc / self.cpu_effeciency
        return mu_T * T_loc + mu_E * E_loc
    
    def generate_choice_index(self) -> Number:
        """Genrate choice for this iteration based on probability vector `self._w`

        Returns:
            Number: The index generated (0 means local computation)
        """
        choice_index = random.choice(len(self.channels) + 1, 1, p=self._w).item() 
        return choice_index

    async def perform_cloud_computation(self, cloud_server: 'CloudServer', channel: Channel, upload_duration: Number):
        env = self.get_current_env()
        await envs.sleep(upload_duration)
        datarate = channel.datarate_between(self, cloud_server)
        transmission_datasize = datarate * upload_duration

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

        # The loop will be executed every `step_interval` seconds
        step_interval = self.get_current_env().g.step_interval
        lr = self.lr
        choice_list = [None] + self.channels
        cloud_server = self._get_cloud_server()

        # Uniform distribution for initialization
        self._w = np.full(channel_num + 1, 1 / (channel_num + 1))
        w = self._w

        # Uses `theta` and `gamma` to be consistent with the notions in paper
        theta = self.active_probability
        gamma = SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR']

        last_transmission: Transmission = None

        while True:
            self._w_history.append(w)
            self.active = True if random.random() < theta else False
            choice_index = self.generate_choice_index()
            self._choice_index = choice_index

            if last_transmission:
                last_transmission.disconnect()
                last_transmission = None

            if self.active:
                if choice_index == 0: # local computation
                    ongoing_trans = None
                    await self.perform_local_computation(step_interval)
                    payoff = self._Q()
                    await envs.wait_for_simul_tasks()
                    
                else: # cloud execution
                    channel: Channel = choice_list[choice_index]
                    last_transmission = channel.connect(self, cloud_server)
                    await self.perform_cloud_computation(cloud_server, channel, step_interval)
                    ongoing_trans = [t.from_node.id for t in channel.transmission_list]
                    payoff = self._I(channel)
                    await envs.wait_for_simul_tasks()

                # `e` is the unit vector
                e = np.zeros(channel_num + 1)
                e[choice_index] = 1

                logger.debug('='*20)
                logger.debug(f'User {self.id}, x: {self._x:.1f} chooses channel {choice_index}, now: {envs.get_current_env().now}')
                logger.debug(f'onging trans {ongoing_trans}')

                logger.debug(w)
                r = 1 - gamma * payoff
                logger.debug(f'index: {choice_index}, payoff: {payoff}, r: {r}, x: {self._x}')
                logger.debug(f'time: {cloud_server.total_compute_time}')
                logger.debug('='*20 + '\n'*2)
                if r < 0:
                    # Set the lower bound of 0
                    logger.warning(f'Genrating r < 0: {r}, gamma: {gamma}, lr: {lr}, choice: {choice_index}, payoff: {payoff}, beta: {_generate_rayleigh_factor(self.id, envs.get_current_env().now)} now: {envs.get_current_env().now} distance: {self._x}')
                    r = 0
                elif r > 1:
                    # Set the lower bound of 1
                    logger.warning(f'Genrating r > 1: {r}, gamma: {gamma}, lr: {lr}, choice: {choice_index}, payoff: {payoff}, beta: {_generate_rayleigh_factor(self.id, envs.get_current_env().now)} distance: {self._x}')
                    r = 1
                new_w = w + lr * r * (e - w)
                
                # To make sure that w is an all-positive vector
                scale_factor = 0.99
                while any(i < 0 for i in new_w):
                    new_w = w + scale_factor * lr * r * (e - w)
                    scale_factor = scale_factor**2
                    logger.warning(f'Genrating w < 0: {new_w}, {w}, choice: {choice_index}, payoff: {payoff}, beta: {_generate_rayleigh_factor(self.id, envs.get_current_env().now)}')

                # To make sure sum(w) equals 1 (Becauses of float number errors it is not always true)
                new_w = new_w / np.sum(new_w)
                self._w = w = new_w
                

                # assert r > 0
            else:
                # If not active, just sleep
                await envs.sleep(step_interval)
                await envs.wait_for_simul_tasks()


class RayleighChannel(Channel):
    """A channel that follows Rayleigh fading. Notice that the datarate will also be affected by connected user. 
    """
    bandwidth: Number = SIMULATION_PARAMETERS['CHANNEL_BANDWITH']
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _get_rayleigh_factor(cls, mobile_id, now):
        beta = _generate_rayleigh_factor(mobile_id, now)
        return beta
    
    @classmethod
    def channel_power(cls, mobile: MobileUser) -> Number:
        """It is the :math:`p_i g_{i,o}` in the original paper.

        Args:
            mobile (MobileUser): Mobile user to be calculated on.
        Returns:
            Number: :math:`p_i g_{i,o}`
        """
        beta = cls._get_rayleigh_factor(mobile.id, envs.get_current_env().now)
        alpha = SIMULATION_PARAMETERS['PATH_LOSS_EXPONENT']
        distance = abs(mobile._x)
        result = mobile.transmit_power * distance**(-alpha) * beta
        # logger.debug(f'chnnel power {result}, beta {beta}')
        return result
    
    def total_channel_power(self, exclude: Optional[MobileUser] = None) -> Number:
        """Total channel_power for active user that are using this channel. :math:`\sum_{i \in \mathcal{A}} p_i g_{i,o}`

        Returns:
            Number: :math:`\sum_{i \in \mathcal{A}} p_i g_{i,o}`
        """
        # TODO: May take irrelevant into account?
        tot_power = 0
        for transmission in self.transmission_list:
            node = transmission.from_node
            assert isinstance(node, Node)
            if node.active and (exclude.id != node.id):
                tot_power += self.channel_power(node)
                
        return tot_power


    def datarate_between(self, mobile: MobileUser, to_node: 'CloudServer') -> Number:
        """Datarate between two nodes. In this case the second node is always remote server.

        Args:
            mobile (MobileUser): Mobile user.
            to_node (CloudServer): Cloud server.

        Returns:
            Number: Datarate between them.
        """
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

class MASLProfile(Profile):
    """This is the class to profile the system.
    """
    def __init__(self, nodes: List[MobileUser], sample_interval: Number) -> None:
        self.nodes = nodes

        # Prepare vectors for system-wide cost calculation
        alpha = SIMULATION_PARAMETERS['PATH_LOSS_EXPONENT']
        self.channel_powers_0 = np.array([node._x**(-alpha) * node.transmit_power for node in nodes]) # channel powers divied by respective rayleigh fading factors

        self._system_wide_cost_samples = []
        self._node_choices = [[] for _ in nodes]
        self._last_sample_ts = None
        super().__init__(sample_interval)
        
    def sample(self):
        logger.info(f'Sampling..., now: {envs.get_current_env().now}')

        now = time.time()
        if self._last_sample_ts is not None:
            logger.info(f'It takes {now - self._last_sample_ts} secs per iteration')
        self._last_sample_ts = now

        logger.debug(f'Channel: 0')
    
        # for node in self.nodes:
        #     cloud_server = envs.get_current_env().g.cloud_server
        #     for channel in node.channels:
        #         logger.info(f'DR for node {node.id} using channel {channel.id}: {channel.datarate_between(node, cloud_server)}')
            
        for node in self.nodes:
            if node._choice_index == 0:
                logger.debug(f'{node.id}')
        channels = self.nodes[0].channels
        for channel in channels:
            logger.debug(f'Channel: {channel.id}')
            for transmission in channel.transmission_list:
                logger.debug(f'{transmission.from_node.id}')
        
        nodes = self.nodes
        for i, node in enumerate(nodes):
            self._node_choices[i].append(node._choice_index)
        result = self.system_wide_cost_vectorized()
        logger.debug(f'cost: {result}')
        self._system_wide_cost_samples.append(result)
    
    def system_wide_cost_vectorized(self):
        epochs = 1000
        betas: np.ndarray = random.standard_exponential((epochs, len(self.nodes)))
        betas = np.ceil(betas * 2) / 2
        # betas = np.ones_like(betas)
        active_probabilities = np.array([node.active_probability for node in self.nodes])
        activeness_v: np.ndarray = random.random((epochs, len(self.nodes))) < active_probabilities
        channel_powers: np.ndarray = self.channel_powers_0 * betas
        assert channel_powers.shape == (epochs, len(self.nodes))
        active_channel_powers = channel_powers * activeness_v
        channels = self.nodes[0].channels
        user_choices = np.empty((epochs, len(self.nodes)), dtype='int')
        # 0 means loal computation
        for i, node in enumerate(self.nodes):
            w = np.array(node._w)
            choices = random.choice(np.arange(len(channels) + 1), epochs, p=w)
            user_choices[:, i] = choices
        
        # column 0 means local computation (always equals to 0)
        total_channel_powers_by_channel = np.zeros((epochs, len(channels) + 1))
        for i in range(1, 1 + len(channels)):
            total_channel_powers_by_channel[:, i] = np.sum(active_channel_powers, axis=1, where=user_choices==i)
        total_channel_powers_by_user = np.empty((epochs, len(self.nodes)))
        for i in range(len(self.nodes)):
            total_channel_powers_by_user[:, i] = total_channel_powers_by_channel[np.arange(epochs), user_choices[:, i]]
        
        B = SIMULATION_PARAMETERS['CHANNEL_BANDWITH']
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']
        temp = 1 + (channel_powers / (total_channel_powers_by_user + sigma_0 - active_channel_powers))
        # assert  np.all((channel_powers / (total_channel_powers_by_user + sigma_0 - active_channel_powers)) > 0)
        datarates: np.ndarray = B * log2(temp)
        avg_datarates = np.mean(datarates, axis=0, where=user_choices!=0)
        total_cost = 0
        print(f'avgdr: , now: {envs.get_current_env().now}')
        for i, node in enumerate(self.nodes):
            if np.isnan(avg_datarates[i]):
                cloud_cost = 0
            else:
                cloud_cost = node.cloud_cost(avg_datarates[i])
            
            total_cost += (node._w[0] * node.local_cost() + sum(node._w[1:]) * cloud_cost) * node.active_probability
        return total_cost

    def system_wide_cost_vectorized_static_choice(self) -> Number:
        """The system wide cost (vectorized version)

        Returns:
            Number: System wide cost.
        """
        epochs = 10000

        betas: np.ndarray = random.standard_exponential((epochs, len(self.nodes)))
        betas = np.ceil(betas * 2) / 2

        active_probabilities = np.array([node.active_probability for node in self.nodes])
        activeness_v: np.ndarray = random.random((epochs, len(self.nodes))) < active_probabilities

        channel_powers: np.ndarray = self.channel_powers_0 * betas
        assert channel_powers.shape == (epochs, len(self.nodes))

        active_channel_powers = channel_powers * activeness_v
        
        channels = self.nodes[0].channels
        channel_nodes = np.zeros((len(channels), len(self.nodes)), dtype='bool')
        for j, node in enumerate(self.nodes):
            i = node._choice_index - 1
            if i >= 0:
                channel_nodes[i][j] = True

        total_channel_powers = np.empty((epochs, len(channels)))
        for i, nodes_bool in enumerate(channel_nodes):
            total_channel_powers[:, i] = np.sum(active_channel_powers, axis=1, where=nodes_bool)

        total_channel_powers_for_nodes = np.zeros_like(channel_powers)
        for i, node in enumerate(self.nodes):
            j = node._choice_index - 1
            if j >= 0:
                total_channel_powers_for_nodes[:, i] = total_channel_powers[:, j]

        B = SIMULATION_PARAMETERS['CHANNEL_BANDWITH']
        sigma_0 = SIMULATION_PARAMETERS['BACKGROUND_NOISE']
        datarates: np.ndarray = B * log2(1 + (channel_powers / (total_channel_powers_for_nodes + sigma_0 - active_channel_powers)))

        avg_datarates = np.average(datarates, axis=0)

        total_cost = 0
        for i, node in enumerate(self.nodes):
            if node._choice_index == 0:
                total_cost += node.active_probability * node.local_cost()
            else:
                total_cost += node.active_probability * node.cloud_cost(avg_datarates[i])
        return total_cost

def create_env(users: List[MobileUser], cloud_server: CloudServer, profile: Optional[MASLProfile], until: Number, step_interval: Number):
    env = SimulationEnvironment(users, until, profile)
    env.g.cloud_server = cloud_server
    env.g.step_interval = step_interval
    return env
    
"""BR-Algorithm for comparison. See `Chen, X., Jiao, L., Li, W., & Fu, X. (2015). Efficient multi-user computation offloading for mobile-edge cloud computing. IEEE/ACM Transactions on Networking, 24(5), 2795-2808.` for more details.
"""
from logging import getLogger
from numbers import Number
import time
from typing import List, Optional

import numpy as np
from numpy import log2, random

from . import envs
from .model import Node, Channel, Profile as ABCProfile, SimulationEnvironment, Transmission
from .masl import SIMULATION_PARAMETERS, RayleighChannel, create_env, MobileUser as MASLMobileUser, Profile as MASLProfile
logger = getLogger(__name__)


class MobileUser(MASLMobileUser):
    def __init__(self, channels: List[RayleighChannel], distance: Optional[Number]=None, active_probability: Optional[Number]=None, run_until: Optional[Number]=None, enable_dueling=False) -> None:
        super().__init__(channels=channels, distance=distance, active_probability=active_probability)

    async def perform_cloud_computation(self, cloud_server: 'CloudServer', channel: Channel, upload_duration: Number, datarate: Number):
        env = self.get_current_env()
        await envs.sleep(upload_duration)
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
    
    def calculate_best_choices(self, current_cost):
        best_choice = None
        best_cost = current_cost
        cloud_server = self.get_current_env().g.cloud_server
        if self.local_cost() < best_cost:
            best_choice = 0
            
        for i, channel in enumerate(self.channels):
            dr = channel.datarate_between(self, cloud_server)
            cost = self.cloud_cost(dr)
            if cost < best_cost:
                best_choice = i + 1
        return best_choice

    async def main_loop(self):
        channel_num = len(self.channels)
        choice_index = random.choice(channel_num + 1, 1).item()
        # choice_index = 0
        cloud_server: CloudServer = self.get_current_env().g.cloud_server
        step_interval = self.get_current_env().g.step_interval
        last_transmission: Optional[Transmission] = None
        while True:
            self._choice_index = choice_index
            self.active = True if random.random() < self.active_probability else False
            if self.active:
                if choice_index == 0:
                    current_cost = self.local_cost()
                    await self.perform_local_computation(step_interval)
                    last_transmission = None
                else:
                    channel = self.channels[choice_index - 1]
                    dr = channel.datarate_between(self, cloud_server)
                    current_cost = self.cloud_cost(dr)
                    last_transmission = channel.connect(self, cloud_server)
                    await self.perform_cloud_computation(cloud_server, channel, step_interval, dr)
                    
                best_choice = self.calculate_best_choices(current_cost=current_cost)
                await envs.wait_for_simul_tasks()
                if last_transmission:
                    last_transmission.disconnect()
                    last_transmission = None

                if best_choice is not None:
                    will_update = (await cloud_server.will_update(True)).value
                else:
                    await cloud_server.will_update(False)
                    will_update = False

                if will_update:
                    choice_index = best_choice
            else:
                if last_transmission:
                    last_transmission.disconnect()
                    last_transmission = None
                await envs.sleep(step_interval)
                await envs.wait_for_simul_tasks()
                await cloud_server.will_update(False)

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

class CloudServer(Node):
    def __init__(self, users: List[MobileUser]) -> None:
        super().__init__()
        self.users = users
        self.requst_to_update_callbacks = []
        self._request_count = 0
        self.cpu_frequency: Number = SIMULATION_PARAMETERS['CLOUD_CPU_CAPABILITY'] # Megacycles/s
    
    async def will_update(self, rtu: bool):
        env = self.get_current_env()
        self._request_count += 1
        if rtu:
            task = env.create_task(None, start=False)
            task.suspend()
            self.requst_to_update_callbacks.append(task)
        else:
            task = env.create_task(None, start=True)
            task.value = False

        if self._request_count == len(self.users):
            if self.requst_to_update_callbacks:
                allowed = random.choice(self.requst_to_update_callbacks, 1).item()
                for t in self.requst_to_update_callbacks:
                    if t is allowed:
                        t.value = True
                    else:
                        t.value = False
                    t.resume()
            self.requst_to_update_callbacks = []
            self._request_count = 0
        
        return await task

class Profile(MASLProfile):
    pass
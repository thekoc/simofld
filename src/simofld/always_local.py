from logging import getLogger
from numbers import Number
from typing import List, Optional

from numpy import log2, random
from . import envs
from .model import Node, Channel, SimulationEnvironment, Transmission
from .masl import SIMULATION_PARAMETERS, RayleighChannel, create_env
from .masl import MobileUser as MASLMobileUser, Profile as MASLProfile
from simofld import masl

logger = getLogger(__name__)


class MobileUser(MASLMobileUser):
    def __init__(self, channels: List[RayleighChannel], distance: Optional[Number]=None, active_probability: Optional[Number]=None, run_until: Optional[Number]=None, enable_dueling=False) -> None:
        super().__init__(channels=channels, distance=distance, active_probability=active_probability)

    async def main_loop(self):
        step_interval = self.get_current_env().g.step_interval
        while True:
            self._choice_index = 0
            await envs.wait_for_simul_tasks()

            self.active = True if random.random() < self.active_probability else False
            if self.active:
                await self.perform_local_computation(step_interval)               
            else:
                await envs.sleep(step_interval)

class CloudServer(masl.CloudServer):
    pass

class Profile(MASLProfile):
    pass
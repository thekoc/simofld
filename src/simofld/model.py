import logging
from numbers import Number
from typing import List, Optional

from . import envs
from .envs import Environment, EnvironmentEntity
from . import utils

logger = logging.getLogger(__name__)

class Data(EnvironmentEntity):
    def __init__(self, size: Number) -> None:
        self.size: Number = size

class LocalData(Data):
    # TODO: Should it be remained?
    def __init__(self, size, owner: 'Node') -> None:
        super().__init__(size)
        self.owner = owner

class Node(EnvironmentEntity):
    def __init__(self) -> None:
        self.total_upload_time = 0
        self.total_download_time = 0
        self.total_compute_time = 0

    async def compute(self, duration: Optional[Number]) -> None:
        """[Coroutine] Do local computation. `self.total_compute_time` will be updated acc

        Args:
            duration (Optional[Number], optional): Duration. Defaults to None.
        """
        await envs.sleep(duration)
        self.total_compute_time += duration

    async def main_loop(self):
        raise NotImplemented

class Transmission(EnvironmentEntity):
    def __init__(self, from_node: Node, to_node: Node, channel: 'Channel') -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.channel = channel
        self.finished = None
        self.started = None
    
    def disconnect(self):
        return self.channel.disconnect(self)

class Channel(EnvironmentEntity):
    def __init__(self) -> None:
        self.transmission_list: List[Transmission] = []

    def datarate_between(self, from_node: Node, to_node: Node) -> Number:
        raise NotImplemented

    def connect(self, from_node: Node, to_node: Node) -> Transmission:
        transmission = Transmission(from_node, to_node, channel=self)
        transmission.finished = False
        transmission.started = True
        self.transmission_list.append(transmission)
        return transmission

    def disconnect(self, transmission: Transmission):
        transmission.finished = True
        self.transmission_list.pop(self.transmission_list.index(transmission))
    
    async def transfer_data(self, from_node: Node, to_node: Node, duration: Number = None, datasize: Number = None):
        """[Coroutine] Transfer data bwteen nodes.

        Args:
            from_node (Node): From.
            to_node (Node): To.
            datasize (Number, optional): Datasize. If this argument has a value, `duration` should be set to `None`. Defaults to None.
            duration (Number, optional): Time limit. If this argument has a value, `datasize` should be set to `None`. Defaults to None.

        Returns:
            [type]: [description] TODO: Decide what to return
        """
        if not utils.singular_not_none(datasize, duration): # Only one of the arguments is allowed to have value.
            raise ValueError('Only one of `datasie`, `duration` is allowed to have a value.')

        dr = self.datarate_between(from_node, to_node)

        assert isinstance(dr, Number)

        if datasize is not None:
            duration = datasize / dr
        else:
            duration = duration
            datasize = duration * dr

        # await envs.wait_for_simul_tasks()
        transmission = Transmission(from_node, to_node, duration=duration)
        self.transmission_list.append(transmission)

        logger.debug(f'start transmission {transmission.id} from {from_node.id} using channel {self.id}, now: {envs.get_current_env().now}')
        await transmission
        from_node.total_upload_time += duration
        to_node.total_download_time += duration
        self.transmission_list.pop(self.transmission_list.index(transmission))
        logger.debug(f'finished transmission {transmission.id} from {from_node.id} using channel {self.id}, now: {envs.get_current_env().now}')

        async def pop_transmission():
            await envs.wait_for_simul_tasks()
            logger.debug(f'finished transmission {transmission.id} from {from_node.id} using channel {self.id}, now: {envs.get_current_env().now}')
            self.transmission_list.pop()
        # envs.get_current_env().create_task(pop_transmission())
        return LocalData(datasize, to_node)

class Profile:
    def __init__(self, sample_interval: Number, no_sample_until=None) -> None:
        self.sample_interval = sample_interval
        self.no_sample_until = no_sample_until

    async def main_loop(self):
        while True:
            await envs.wait_for_simul_tasks()
            if self.no_sample_until is None or envs.get_current_env().now >= self.no_sample_until:
                self.sample()
            await envs.sleep(self.sample_interval)

    def sample(self):
        raise NotImplemented

class SimulationEnvironment(Environment):
    def __init__(self, nodes: List[Node], until: Number, profile: Optional[Profile] = None) -> None:
        coros = [node.main_loop() for node in nodes]
        p_list = [0 for _ in nodes]
        if profile:
            coros += [profile.main_loop()]
            p_list += [1]
        super().__init__(coros, priority_list=p_list, initial_time=0, until=until)
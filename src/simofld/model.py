import logging
from numbers import Number
from typing import List, Optional
from weakref import WeakSet


from . import envs
from . import utils

logger = logging.getLogger(__name__)

class EnvironmentEntityMeta(type):
    def __new__ (cls, name, bases, namespace):
        namespace['_instances'] = WeakSet()
        return super().__new__(cls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._instances.add(instance)
        return instance
    
    @property
    def instances(cls):
        return [x for x in cls._instances]

class EnvironmentEntity(metaclass=EnvironmentEntityMeta):
    def get_current_env(self):
        return envs.get_current_env()

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

class Transmission(envs.Task):
    def __init__(self, from_node: Node, to_node: Node, duration: Number) -> None:
        super().__init__(coro=envs.sleep(duration), wait_until=None, callbacks=[])
        self.from_node = from_node
        self.to_node = to_node


class Channel(EnvironmentEntity):
    def __init__(self) -> None:
        self.transmission_list: List[Transmission] = []

    def datarate_between(self, from_node: Node, to_node: Node) -> Number:
        raise NotImplemented
    
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

        transmission = Transmission(from_node, to_node, duration=duration)
        self.transmission_list.append(transmission)

        await transmission
        from_node.total_upload_time += duration
        to_node.total_download_time += duration

        self.transmission_list.pop()
        return LocalData(datasize, to_node)

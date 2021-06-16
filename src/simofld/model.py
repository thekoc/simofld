import queue
from . import envs
from . import utils
from numbers import Number
from typing import List, Optional

class EnvironmentEntity:
    def current_env(self):
        return envs.get_current_env()

class Data(EnvironmentEntity):
    def __init__(self, size) -> None:
        self.size = size

class LocalData(Data):
    def __init__(self, size, owner: 'Node') -> None:
        super().__init__(size)
        self.owner = owner


class Node(EnvironmentEntity):
    def __init__(self, data_process_rate: Optional[Number] = None) -> None:
        self.data_process_rate = data_process_rate
        self.total_upload_time = 0
        self.total_download_time = 0
        self.total_compute_time = 0

    async def compute(self, duration: Optional[Number] = None, datasize: Optional[Number] = None):
        if not utils.singular_not_none(datasize, duration):
            raise ValueError('Only one of `datasie`, `duration` is allowed to have a value.')

        if datasize is not None:
            if self.data_process_rate is None or self.data_process_rate <= 0:
                raise ValueError('`data_process_rate` must be a positive number')

            duration = datasize / self.data_process_rate

        await envs.sleep(duration)
        self.total_compute_time += duration

    async def main_loop(self):
        pass

class Channel(EnvironmentEntity):
    def __init__(self) -> None:
        self.ongoing_transmission_num = 0

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
            [type]: [description]
        """
        if not utils.singular_not_none(datasize, duration): # Only one of the arguments is allowed to have value.
            raise ValueError('Only one of `datasie`, `duration` is allowed to have a value.')

        self.ongoing_transmission_num += 1

        dr = self.datarate_between(from_node, to_node)
        assert isinstance(dr, Number)

        if datasize is not None:
            duration = datasize / dr
        else:
            duration = duration
            datasize = duration * dr


        await envs.sleep(duration)
        from_node.total_upload_time += duration
        to_node.total_download_time += duration

        self.ongoing_transmission_num -= 1
        return LocalData(datasize, to_node)

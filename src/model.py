import envs
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

    async def compute(self, datasize: Number):
        if self.data_process_rate is None or self.data_process_rate <= 0:
            raise ValueError('`data_process_rate` must be a number larger than 0')

        time_cost = datasize / self.data_process_rate
        await envs.sleep(time_cost)
    
    async def main_loop(self):
        pass

class Channel(EnvironmentEntity):
    def __init__(self) -> None:
        self.ongoing_transmission_num = 0

    def datarate_between(self, from_node: Node, to_node: Node) -> Number:
        raise NotImplemented
    
    async def transfer_data(self, from_node: Node, to_node: Node, datasize: Number = None, time_limit: Number = None):
        assert datasize is None != time_limit is None # can be constrained with only one of the arguments 
        self.ongoing_transmission_num += 1

        dr = self.datarate_between(from_node, to_node)
        if datasize is not None:
            time_cost = datasize / dr
        else:
            time_cost = time_limit
            datasize = time_cost * dr


        await envs.sleep(time_cost)
        return LocalData(datasize, to_node)

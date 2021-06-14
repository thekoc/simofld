import envs
from numbers import Number
from typing import List, Optional


class Data:
    def __init__(self, size) -> None:
        self.size = size

class LocalData(Data):
    def __init__(self, size, owner: 'Node') -> None:
        super().__init__(size)
        self.owner = owner


class Node:
    def __init__(self, data_process_rate: Optional[Number] = None) -> None:
        self.data_process_rate = data_process_rate
    
    async def tansfer_data(self, to_node: 'Node', channel: 'Channel', datasize: Number = None, time_limit: Number = None):
        assert datasize is None != time_limit is None # can be constrained with only one of the arguments 
        return await channel.tranfer_data(self, to_node, datasize=datasize, time_limit=time_limit)

    async def compute(self, datasize: Number):
        if self.data_process_rate is None or self.data_process_rate <= 0:
            raise ValueError('`data_process_rate` must be a number larger than 0')

        time_cost = datasize / self.data_process_rate
        await envs.sleep(time_cost)
    
    async def main_loop(self):
        pass

class Channel:
    def __init__(self) -> None:
        self.ongoing_transmission_num = 0

    def datarate_between(self, from_node: Node, to_node: Node) -> Number:
        raise NotImplemented
    
    async def transfer_data(self, from_node: Node, to_node: Node, datasize: Number = None, time_limit: Number = None):
        assert datasize is None != time_limit is None # can be constrained with only one of the arguments 
        self.ongoing_transmission_num += 1
        if datasize is not None:
            dr = self.datarate_between(from_node, to_node)
            time_cost = datasize / dr
        else:
            time_cost = time_limit

        await envs.sleep(time_cost)



class MobileUser(Node):
    def __init__(self, data_process_rate: Optional[Number], channels: List[Channel]) -> None:
        super().__init__(data_process_rate=data_process_rate)


class CloudServer(Node):
    pass
import env
from numbers import Number


class Data:
    def __init__(self, size) -> None:
        self.size = size

class Node:
    def __init__(self) -> None:
        pass
    
    async def tansfer(self, to_node: 'Node', datasize: Number, channel: 'Channel', time_limit=None):
        dr = channel.datarate_between(self, to_node)
        time_cost = datasize / dr
        await env.sleep(time_cost)

    async def compute(self, data):
        pass

class Channel:
    def __init__(self) -> None:
        pass

    def datarate_between(self, node_a: Node, node_b: Node) -> Number:
        raise NotImplemented




class MobileUser(Node):
    pass

class CloudServer(Node):
    pass
from numbers import Number

class Data:
    def __init__(self, size) -> None:
        self.size = size

class Node:
    def __init__(self) -> None:
        pass
    
    async def tansfer(self, node_to: 'Node', data_size: Number, channel: 'Channel'):
        pass

class Channel:
    def __init__(self) -> None:
        pass

    def data_rate_between(self, node_a: Node, node_b: Node) -> Number:
        raise NotImplemented




class MobileUser(Node):
    pass

class CloudServer(Node):
    pass
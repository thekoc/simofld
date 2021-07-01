
import sys
sys.path.insert(0, 'src')

import unittest
import random
from simofld.masl import CloudServer, MobileUser, RayleighChannel
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils
from numpy import random as np_random
import numpy as np
from matplotlib import pyplot as plt

# np_random.seed(5)
# random.seed(1)


class TestMASL(unittest.TestCase):
    def test_main_algorithm(self):
        channels = [RayleighChannel() for _ in range(6)]
        nodes = [MobileUser(channels) for _ in range(6)]
        cloud_server = CloudServer()
        with envs.create_env([node.main_loop() for node in nodes], until=10) as env:
            env.g.cloud_server = cloud_server
            env.run()
        
        for node in nodes:
            print(node._w.round(1) * 10)
        
        fig, ax = plt.subplots()
        for i in range(len(channels) + 1):
            ax.plot([w[i] for w in nodes[0]._w_history])
        plt.show()

if __name__ == '__main__':
    unittest.main()
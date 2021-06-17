
import sys
sys.path.insert(0, 'src')

import unittest
import random
from simofld.masl import CloudServer, MobileUser, RayleighChannel
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils
from numpy import random as np_random

np_random.seed(5)
random.seed(1)


class TestMASL(unittest.TestCase):
    def test_main_algorithm(self):
        channels = [RayleighChannel() for _ in range(1)]
        nodes = [MobileUser(channels) for _ in range(3)]
        cloud_server = CloudServer()
        with envs.create_env([node.main_loop() for node in nodes], until=500) as env:
            env.g.cloud_server = cloud_server
            env.run()

if __name__ == '__main__':
    unittest.main()
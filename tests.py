import unittest
import random
import sys
sys.path.insert(0, 'src')
from simofld import envs, exceptions
from simofld.model import Channel, Node


class TestEventLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.assertRaises(exceptions.NoCurrentEnvironmentError, envs.get_current_env)

    def test_envs(self):
        pass

    def test_sleep(self):
        async def func():
            self.assertEqual(envs.get_current_env().now, 0)
            delay = 0.5
            await envs.sleep(delay)
            self.assertEqual(envs.get_current_env().now, delay)

        with envs.create_env([func()]) as env:
            env.run()

    def test_gather(self):
        async def func():
            self.assertEqual(envs.get_current_env().now, 0)
            random.seed(0)
            delays = [random.random() for _ in range(10)]
            await envs.gather([envs.sleep(delay) for delay in delays])
            self.assertEqual(max(delays), envs.get_current_env().now)

        with envs.create_env([func()]) as env:
            env.run()

class TestModel(unittest.TestCase):
    def test_channel(self):
        channel = Channel()
        channel.datarate_between = lambda a, b: 1
        node_a, node_b = Node(), Node()
        channel.transfer_data

if __name__ == '__main__':
    unittest.main()
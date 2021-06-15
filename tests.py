import unittest
import random
import sys
sys.path.insert(0, 'src')
from simofld import envs, exceptions
from simofld.model import Channel, Node

random.seed(0)

class TestEventLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.assertRaises(exceptions.NoCurrentEnvironmentError, envs.get_current_env)

    def test_envs(self):
        pass

    def test_sleep(self):
        async def coro():
            self.assertEqual(envs.get_current_env().now, 0)
            delay = 0.5
            await envs.sleep(delay)
            self.assertEqual(envs.get_current_env().now, delay)

        with envs.create_env([coro()]) as env:
            env.run()

    def test_gather(self):
        async def coro():
            self.assertEqual(envs.get_current_env().now, 0)
            delays = [random.random() for _ in range(10)]
            await envs.gather([envs.sleep(delay) for delay in delays])
            self.assertEqual(max(delays), envs.get_current_env().now)

        with envs.create_env([coro()]) as env:
            env.run()

class TestModel(unittest.TestCase):
    def test_channel(self):
        async def coro():
            channel = Channel()
            channel.datarate_between = lambda *_: 1
            node_a, node_b = Node(), Node()
            duration = random.random()
            await channel.transfer_data(node_a, node_b, time_limit=duration)
            self.assertEqual(duration, envs.get_current_env().now)


        with envs.create_env([coro()]) as env:
            env.run()

if __name__ == '__main__':
    unittest.main()
import unittest
import random
import sys
sys.path.insert(0, 'src')
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils

random.seed(0)

class TestEventLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.assertRaises(exceptions.NoCurrentEnvironmentError, envs.get_current_env)

    def test_envs(self):
        # test lifespan

        async def coro():

            self.assertEqual(envs.get_current_env().now, 0)
            delay = random.random()
            task = envs.get_current_env().create_task(envs.sleep(delay))
            x = await task

            self.assertEqual(envs.get_current_env().now, delay)

        with envs.create_env([coro()]) as env:
            env.run()

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

            dr = random.random()
            channel.datarate_between = lambda *_: dr
            node_a, node_b = Node(), Node()

            # test argument checking
            with self.assertRaises(ValueError):
                await channel.transfer_data(node_a, node_b, duration=1, datasize=1)
                await channel.transfer_data(node_a, node_b, duration=None, datasize=None)
            
            total_duration = 0
            
            # test duration
            duration = random.random() + 0.1
            total_duration += duration
            now = envs.get_current_env().now
            await channel.transfer_data(node_a, node_b, duration=duration)
            self.assertEqual(duration, envs.get_current_env().now - now)

            # test data_size
            datasize = random.random() + 0.1
            now = envs.get_current_env().now
            duration = datasize / dr
            total_duration += duration
            await channel.transfer_data(node_a, node_b, datasize=datasize)
            self.assertEqual(duration, envs.get_current_env().now - now)

            # test total duration
            duration = random.random()
            channel.transfer_data(node_b, node_a, duration=duration)
            self.assertEqual(node_a.total_upload_time, total_duration)
            self.assertEqual(node_a.total_download_time, duration)
            
            self.assertEqual(node_b.total_download_time, total_duration)
            self.assertEqual(node_b.total_upload_time, duration)


        with envs.create_env([coro()]) as env:
            env.run()

class TestUtils(unittest.TestCase):
    def test_utils(self):
        self.assertTrue(utils.singular_not_none(1))
        self.assertTrue(utils.singular_not_none(1, None))
        self.assertTrue(utils.singular_not_none(1, None, None))

        self.assertFalse(utils.singular_not_none(1, 1, None))
        self.assertFalse(utils.singular_not_none(None, None, None))
        self.assertFalse(None)

if __name__ == '__main__':
    unittest.main()
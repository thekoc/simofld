
from queue import Queue
import sys
sys.path.insert(0, 'src')

import unittest
import random
from simofld.masl import CloudServer, MobileUser, RayleighChannel
from simofld import envs, exceptions
from simofld.model import Channel, Node
import simofld.utils as utils
from numpy import random as np_random
np_random.seed(1)
random.seed(1)

class TestEventLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.assertRaises(exceptions.NoCurrentEnvironmentError, envs.get_current_env)

    def test_envs(self):
        # test lifespan

        async def coro():
            self.assertEqual(envs.get_current_env().now, 0)
            delay = random.random()
            task = await envs.get_current_env().create_task(envs.sleep(delay))
            self.assertEqual(task.lifespan, (0, delay))

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
            delays = [random.random() for _ in range(1)]
            await envs.gather([envs.sleep(delay) for delay in delays])
            self.assertEqual(max(delays), envs.get_current_env().now)

        with envs.create_env([coro()]) as env:
            env.run()
    
    def test_keep_order(self):
        q = []
        async def add_queue(i):
            q.append(i)

        async def coro():
            env = envs.get_current_env()
            for i in range(100):
                env.create_task(add_queue(i))

            await envs.sleep(1)
            self.assertEqual(q, list(range(100)))
        with envs.create_env([coro()]) as env:
            env.run()
    
    def test_wait_others(self):
        q = []
        async def add_queue(i):
            if i == 0:
                await envs.wait_for_simul_tasks()
            q.append(i)

        async def coro():
            env = envs.get_current_env()
            for i in range(100):
                env.create_task(add_queue(i))

            await envs.sleep(1)
            self.assertEqual(q, list(range(1, 100)) + [0])
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
            self.assertAlmostEqual(duration, envs.get_current_env().now - now)

            # test data_size
            datasize = random.random() + 0.1
            now = envs.get_current_env().now
            duration = datasize / dr
            total_duration += duration
            await channel.transfer_data(node_a, node_b, datasize=datasize)
            self.assertAlmostEqual(duration, envs.get_current_env().now - now)

            # test total duration
            duration = random.random()
            await channel.transfer_data(node_b, node_a, duration=duration)
            self.assertAlmostEqual(node_a.total_upload_time, total_duration)
            self.assertAlmostEqual(node_a.total_download_time, duration)

            self.assertAlmostEqual(node_b.total_download_time, total_duration)
            self.assertAlmostEqual(node_b.total_upload_time, duration)

            # test ongoing_transmission_num
        with envs.create_env([coro()]) as env:
            env.run()

    def test_entity_count(self):
        with envs.create_env():
            self.assertEqual(len(Node.instances), 0)
            def f():
                self.assertEqual(len(Node.instances), 0)
                node_a = Node()
                self.assertEqual(len(Node.instances), 1)
                node_b = Node()
                self.assertEqual(len(Node.instances), 2)
                Node()
                self.assertEqual(len(Node.instances), 2)
            f()
            self.assertEqual(len(Node.instances), 0)

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
import unittest
import random

import src.envs as envs
import src.exceptions as exceptions


class TestEventLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.assertRaises(exceptions.NoCurrentEnvironmentError, envs.get_active_env)

    def test_envs(self):
        pass

    def test_sleep(self):
        async def func():
            self.assertEqual(envs.get_active_env().now, 0)
            delay = 0.5
            await envs.sleep(delay)
            self.assertEqual(envs.get_active_env().now, delay)

        with envs.create_env([func()]) as env:
            env.run()

    def test_gather(self):
        async def func():
            self.assertEqual(envs.get_active_env().now, 0)
            random.seed(0)
            delays = [random.random() for _ in range(10)]
            await envs.gather([envs.sleep(delay) for delay in delays])
            self.assertEqual(max(delays), envs.get_active_env().now)

        with envs.create_env([func()]) as env:
            env.run()

if __name__ == '__main__':
    unittest.main()
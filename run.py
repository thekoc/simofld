import asyncio
from src import envs
import random
from src.envs import Environment, sleep, gather, get_active_env, create_env

def test_sleep():
    async def aprint(s):
        await sleep(1)
        print(s)
        
    async def func():
        print('starting...')
        await sleep(0.5)
        print(f'ending, current time {get_active_env().now}...')

    with create_env([func()]) as env:
        env.run()

def test_gather():
    async def func():
        print('starting...')
        random.seed(1)
        delays = [random.random() for _ in range(2)]
        await envs.gather([envs.sleep(delay) for delay in delays])
        pass

    with envs.create_env([func()]) as env:
        env.run()

if __name__ == '__main__':
    test_sleep()
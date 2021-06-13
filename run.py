import asyncio
from src.envs import Environment, sleep, gather, get_active_env, create_env

def test_sleep():
    async def func():
        print('starting...')
        await sleep(0.1)
        print('ending...')

    with create_env([func()]) as env:
        env.run()

if __name__ == '__main__':
    test_sleep()
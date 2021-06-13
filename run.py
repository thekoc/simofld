import asyncio
from src.env import Environment, sleep, gather, get_active_env, create_env

async def  delayed_print(delay):
    await sleep(delay)
    print('xxx')

async def func(delay):
    print(f'start, current time: {get_active_env().now}')
    await sleep(delay)
    print(f'sleeped, now {get_active_env().now}')
    # await delayed_print(3)
    await gather(delayed_print(1), delayed_print(5))
    print(f'end, current time: {get_active_env().now}')

if __name__ == '__main__':
    with create_env([func(1)], 0) as env:
        env.run()
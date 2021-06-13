from numbers import Number
import types
from typing import Optional, Union, NoReturn, List
from . import exceptions

import heapq
from typing import Coroutine
from contextlib import contextmanager
from functools import wraps


OptionalCoroutine = Union[Coroutine, None]
OptionalEnvironment = Union['Environment', None]
OptionalTask = Union['Task', None]

TaskList = List['Task']
CoroutineList = List[Coroutine]

class Task:
    def __init__(self, coro: Coroutine, wait_until=None, callbacks=None):
        self.coro = coro # type: Coroutine
        self.wait_until = wait_until
        self.callbacks = callbacks if callbacks else []
    
    def step(self):
        if self.coro is None:
            return
        else:
            next_task = self.coro.send(None)
            if next_task:
                next_task.callbacks.append(self.step)
    
    def __await__(self):
        yield self

    def __eq__(self, other):
        return self.wait_until == other.wait_until

    def __lt__(self, other):
        return self.wait_until < other.wait_until

def get_active_env():
    if Environment._activae_env is not None:
        return Environment._activae_env
    else:
        raise exceptions.NoCurrentEnvironmentError

def get_active_task():
    return get_active_env()._active_task

class EmptyTask(Task):
    def __init__(self, wait_until=None, callbacks=None):
        super().__init__(None, wait_until=wait_until, callbacks=callbacks)


class Environment:
    _activae_env: OptionalEnvironment = None

    def __init__(self, coros, initial_time=0) -> None:
        self.now = initial_time
        self._coros = coros
        self._active_task: OptionalTask = None
        self._running_tasks: TaskList = [Task(coro, wait_until=self.now) for coro in coros] # type: list[Task]
        self.prev_env: OptionalEnvironment = None

    def start_task(self, task: Task, delay: Optional[Number] = None):
        if delay is None:
            task.wait_until = self.now
        else:
            task.wait_until = self.now + delay

        heapq.heappush(self._running_tasks, task)
        return task
    
    def create_task(self, coro: Coroutine, wait_until=None, callbacks=None, start=True):
        task = Task(coro=coro, wait_until=wait_until, callbacks=callbacks)
        if start:
            self.start_task(task)
        return task

    def run(self):
        while self._running_tasks:
            task = heapq.heappop(self._running_tasks)
            self.now = task.wait_until
            self._active_task = task
            try:
                task.step()
                if task.callbacks:
                    for callback in task.callbacks:
                        callback()
            except StopIteration:
                print('Task finished')
            else:
                self._active_task = None

        print(f'all tasks done, current time: {self.now}')

    def __enter__(self):
        self.prev_env = Environment._activae_env
        Environment._activae_env = self
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        Environment._activae_env = self.prev_env


def create_env(coros: List[Coroutine], initial_time: Number = 0):
    return Environment(coros, initial_time)


async def sleep(delay, env: 'Environment' = None):
    if env is None:
        env = get_active_env()
    
    return await env.start_task(EmptyTask(), delay)


async def gather(*coros, env: 'Environment' = None):
    if env is None:
        env = get_active_env()
    coro_num = len(coros)
    done_num = 0
    gathering_task = EmptyTask()
    def resume():
        nonlocal done_num
        done_num += 1
        if done_num == coro_num:
            env.start_task(gathering_task)
    
    for coro in coros:
        task = Task(coro, callbacks=[resume])
        env.start_task(task)

    return gathering_task
    

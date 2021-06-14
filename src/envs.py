import heapq
from numbers import Number
from typing import Optional, List, Coroutine

from . import exceptions



class Task:
    def __init__(self, coro: Coroutine, wait_until=None, callbacks=None):
        self.coro = coro # type: Coroutine
        self._done = False
        self.wait_until = wait_until
        self.callbacks = callbacks if callbacks else []
    
    def step(self):
        env = get_current_env()
        if self.coro is None:
            return
        else:
            try:
                next_task = self.coro.send(None)
                if next_task:
                    def callback():
                        return env.start_task(self)
                    next_task.callbacks.append(callback)
            except StopIteration:
                self._done = True
    
    def done(self):
        return self._done
    
    def __await__(self):
        yield self

    def __eq__(self, other):
        return self.wait_until == other.wait_until

    def __lt__(self, other):
        return self.wait_until < other.wait_until

def get_current_env():
    if Environment._current_env is not None:
        return Environment._current_env
    else:
        raise exceptions.NoCurrentEnvironmentError

def get_active_task():
    return get_current_env()._active_task

class EmptyTask(Task):
    def __init__(self, wait_until=None, callbacks=None):
        super().__init__(None, wait_until=wait_until, callbacks=callbacks)

class Environment:
    _current_env: Optional['Environment'] = None

    def __init__(self, coros, initial_time=0) -> None:
        self.now = initial_time
        self._coros = coros
        self._active_task: Optional[Task] = None
        self._running_tasks: List[Task] = [Task(coro, wait_until=self.now) for coro in coros] # type: list[Task]
        self.prev_env: Optional[Environment] = None

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
            task.step()
            if task.done():
                print('Task done')
                if task.callbacks:
                    for callback in task.callbacks:
                        callback()
            self._active_task = None

        print(f'all tasks done, current time: {self.now}')

    def __enter__(self):
        self.prev_env = Environment._current_env
        Environment._current_env = self
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        Environment._current_env = self.prev_env


def create_env(coros: List[Coroutine], initial_time: Number = 0):
    return Environment(coros, initial_time)


async def sleep(delay, env: 'Environment' = None):
    if env is None:
        env = get_current_env()
    
    return await env.start_task(EmptyTask(), delay)


async def gather(coros, env: 'Environment' = None):
    if env is None:
        env = get_current_env()
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

    return await gathering_task

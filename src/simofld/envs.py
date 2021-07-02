import logging
import sys
from queue import PriorityQueue
from numbers import Number
from typing import Callable, Optional, List, Coroutine
from weakref import WeakSet

from . import exceptions

TASK_PRIORITY_MAX_NUMBER = 9999999

logger = logging.getLogger(__name__)

class EnvironmentEntityMeta(type):
    def __new__ (cls, name, bases, namespace):
        namespace['_instances'] = WeakSet()
        namespace['_count'] = 1
        return super().__new__(cls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._instances.add(instance)
        instance._id = cls._count
        cls._count += 1
        return instance
    
    @property
    def instances(cls):
        return [x for x in cls._instances]

class EnvironmentEntity(metaclass=EnvironmentEntityMeta):
    def get_current_env(self):
        return get_current_env()
    
    @property
    def id(self) -> Number:
        return self._id

class EnvironmentStorage:
    pass

class Task(EnvironmentEntity):
    def __init__(self, coro: Optional[Coroutine], priority: Number = 0, wait_until: Optional[Number] = None, callbacks: Optional[List[Callable]] = []):
        """Init a task. Don't call it directly.

        Args:
            coro (Optional[Coroutine]): Coroutine inside the task.
            wait_until (Optional[Number], optional): When should the task be scheduled. If None, the task will be executed once started. Defaults to None.
            callbacks (Optional[List[Callable]], optional): Set it to None if return to the parent coroutine is not wanted. Defaults to [].
        """
        self.coro = coro # type: Coroutine
        self.lifespan = None
        self.priority = priority
        self._done = False
        self._started = False
        self._suspended = False
        self.wait_until = wait_until
        self.callbacks = [] if callbacks == [] else callbacks
    
    def step(self):
        env = get_current_env()
        if self.coro is None:
            self._done = True
        else:
            try:
                next_task: Task = self.coro.send(None)
                if not isinstance(next_task, Task):
                    raise TypeError('Yield value must be a `Task`')
                if next_task.callbacks is not None:
                    def callback():
                        return env.start_task(self)
                    next_task.callbacks.append(callback)
            except StopIteration:
                self._done = True
    
    def suspend(self):
        self._suspended = True

    def resume(self):
        get_current_env().start_task(self)
        self._suspended = False
    
    @property
    def done(self) -> bool:
        return self._done
    
    def __await__(self):
        env = get_current_env()
        if not self._started and not self._suspended:
            env.start_task(self)

        start = env.now
        yield self
        self.lifespan = (start, env.now)
        return self

    def __lt__(self, other: 'Task'):
        if self.wait_until == other.wait_until:
            if self.priority == other.priority:
                return self.id < other.id
            else:
                return self.priority < other.priority
        if self.wait_until is None:
            return True
        elif other.wait_until is None:
            return False
        else:
            return self.wait_until < other.wait_until

def get_current_env():
    if Environment._current_env is not None:
        return Environment._current_env
    else:
        raise exceptions.NoCurrentEnvironmentError

def get_active_task():
    return get_current_env()._active_task

class Environment:
    _current_env: Optional['Environment'] = None

    def __init__(self, coros: List[Coroutine], initial_time: Number = 0, until: Optional[Number] = None, priority_list=None) -> None:
        self.now: Number = initial_time
        self.until: Optional[Number] = until
        self._coros = coros
        self._priority_list = priority_list
        self._active_task: Optional[Task] = None
        self._running_tasks = PriorityQueue() # Initialized using coros in self.run()
        self.prev_env: Optional[Environment] = None
        self.g = EnvironmentStorage()

    def start_task(self, task: Task):
        self._running_tasks.put(task)
        task._started = True
        return task
    
    def create_task(self, coro: Coroutine, delay: Optional[Number] = 0, callbacks: Optional[List[Callable]] = [], start: bool = True, priority: Number = 0):
        task = Task(coro=coro, priority=priority, wait_until=self.now + delay, callbacks=callbacks)
        if start:
            self.start_task(task)
        return task

    def run(self):
        if self._priority_list:
            for coro, priority in zip(self._coros, self._priority_list):
                self.create_task(coro=coro, priority=priority)
        else:
            for coro in self._coros:
                self.create_task(coro)
        
        while not self._running_tasks.empty():
            if self.until and self.now > self.until:
                break
            task: Task = self._running_tasks.get_nowait()

            if task._suspended:
                continue
                
            if task.wait_until is not None:
                self.now = max(task.wait_until, self.now)
            self._active_task = task
            task.step()
            logger.debug(f'Task step, now {self.now}')

            if task.done:
                logger.debug(f'Task done, now {self.now}')
                if task.callbacks:
                    for callback in task.callbacks:
                        callback()
            self._active_task = None

        logger.debug(f'all tasks done, current time: {self.now}')

    def __enter__(self):
        self.prev_env = Environment._current_env
        Environment._current_env = self
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        Environment._current_env = self.prev_env


def create_env(coros: Optional[List[Coroutine]]=None, initial_time: Number = 0, until: Optional[Number] = None):
    return Environment(coros if coros else [], initial_time, until)


async def sleep(delay, env: 'Environment' = None):
    if env is None:
        env = get_current_env()

    return await env.create_task(None, delay)

async def wait_for_simul_tasks(env: 'Environment' = None):
    """Wait for other tasks that are scheduled for the same time.
    """
    if env is None:
        env = get_current_env()
    return await env.create_task(coro=None, delay=0, priority=TASK_PRIORITY_MAX_NUMBER)


async def gather(coros, env: 'Environment' = None):
    if env is None:
        env = get_current_env()

    gathering_task = env.create_task(None, start=False)

    coro_num = len(coros)
    done_num = 0

    def resume():
        nonlocal done_num
        done_num += 1
        if done_num == coro_num:
            gathering_task.resume()
    
    for coro in coros:
        env.create_task(coro, callbacks=[resume])
    gathering_task.suspend()

    return await gathering_task

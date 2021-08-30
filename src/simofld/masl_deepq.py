from logging import getLogger
from numbers import Number
import random as py_random
from typing import List, Optional
from collections import deque

import numpy as np
from numpy import log2, random
from tensorflow import keras

from . import envs
from .model import Node, Channel, SimulationEnvironment, Transmission
from .masl import SIMULATION_PARAMETERS, RayleighChannel, create_env
from .masl import MobileUser as MASLMobileUser, MASLProfile as MASLProfile
from simofld import masl
logger = getLogger(__name__)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return py_random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = py_random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class MobileUser(MASLMobileUser):
    def __init__(self, channels: List[RayleighChannel], distance: Optional[Number]=None, active_probability: Optional[Number]=None) -> None:
        super().__init__(channels=channels, distance=distance, active_probability=active_probability)
        self.channels = channels
        self._x = distance 
        self.transmit_power = SIMULATION_PARAMETERS['TRANSMIT_POWER'] # p_i
        self.active = None
        self.cpu_frequency = SIMULATION_PARAMETERS['LOCAL_CPU_CAPABILITY'][0] # Megacycles/s
        self.cpu_effeciency = SIMULATION_PARAMETERS['COMPUTING_ENERGY_EFFECIENCY'][0] # Megacycles/J TODO: Random selection
        self.active_probability = active_probability
        self._datasize = SIMULATION_PARAMETERS['DATA_SIZE']
        self._choice_index = None
        self._payoff_weight_energy = 0
        self._payoff_weight_time = 1 - self._payoff_weight_energy

        self.dqn_agent = DQNAgent(state_size=len(self.channels) + 1, action_size=len(self.channels) + 1)

    def reward(self, choice_index):
        if choice_index == 0:
            payoff = self._Q()
        else:
            channel = self.channels[choice_index - 1]
            payoff = self._I(channel)
        
        r = - 1e5 * payoff
        return r


    def get_state(self):
        state = [self._Q()] + [self._I(channel) for channel in self.channels]
        return np.array(state).reshape((1, len(state)))

    async def main_loop(self):
        last_state = None
        batch_size = 32

        last_transmission: Optional[Transmission] = None
        cloud_server = self._get_cloud_server()
        step_interval = self.get_current_env().g.step_interval
        self._choice_index = random.randint(0, len(self.channels) + 1)
        while True:
            self.active = True if random.random() < self.active_probability else False
            if last_transmission:
                last_transmission.disconnect()
                last_transmission = None

            if self.active:
                if self._choice_index == 0:
                    await self.perform_local_computation(step_interval)
                else:
                    channel = self.channels[self._choice_index - 1]
                    last_transmission = channel.connect(self, cloud_server)
                    await self.perform_cloud_computation(cloud_server, channel, step_interval)
                
                state = self.get_state()
                reward = self.reward(self._choice_index)
                await envs.wait_for_simul_tasks()
                
                if last_state is not None:
                    action = self._choice_index
                    self.dqn_agent.memorize(last_state, action, reward, state, False)
                last_state = state
                
                if len(self.dqn_agent.memory) > batch_size:
                    self.dqn_agent.replay(batch_size)

class CloudServer(masl.CloudServer):
    pass

class Profile(MASLProfile):
    def sample(self):
        print(f'now: {envs.get_current_env().now}')
        return super().sample()
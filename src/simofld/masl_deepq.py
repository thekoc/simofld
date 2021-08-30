from logging import getLogger
from numbers import Number
import random as py_random
from typing import List, Optional
from collections import deque

import numpy as np
from numpy import log2, random
from tensorflow import keras
import tensorflow as tf

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
        self.gamma = 0.5    # discount rate
        self.alpha = 0.7
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        init = tf.keras.initializers.he_uniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return py_random.randrange(self.action_size)
        act_values = self.model.predict([state])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = py_random.sample(self.memory, batch_size)

        current_states = np.array([e[0] for e in minibatch])
        current_q_list = self.model.predict(current_states)
        
        next_states = [e[3] for e in minibatch]
        future_max_q_list = np.max(self.model.predict(next_states), axis=1)
        
        X = []
        Y = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            future_max_q = future_max_q_list[i]
            current_q = current_q_list[i]
            if not done:
                target = (reward + self.gamma *
                          future_max_q)
            else:
                target = reward
            current_q[action] = (1 - self.alpha) * current_q[action] + self.alpha * target
            X.append(state)
            Y.append(current_q)

        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
        print(f'epsilon: {self.epsilon}')
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
        return state

    async def main_loop(self):
        last_state = None
        batch_size = 32

        last_transmission: Optional[Transmission] = None
        cloud_server = self._get_cloud_server()
        step_interval = self.get_current_env().g.step_interval
        self._choice_index = random.randint(0, len(self.channels) + 1)
        epoch = 0
        while True:
            self.active = True if random.random() < self.active_probability else False
            if last_transmission:
                last_transmission.disconnect()
                last_transmission = None

            if self.active:
                epoch += 1
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
                self._choice_index = self.dqn_agent.act(state)
                
                
                if len(self.dqn_agent.memory) > batch_size:
                    if epoch % 4 == 0:
                        self.dqn_agent.replay(batch_size)

class CloudServer(masl.CloudServer):
    pass

class Profile(MASLProfile):
    def sample(self):
        print(f'now: {envs.get_current_env().now}')
        return super().sample()
from logging import getLogger
from numbers import Number
import random as py_random
from typing import List, Optional
from collections import deque

import numpy as np
from numpy import log2, random
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from . import envs
from .model import Node, Channel, SimulationEnvironment, Transmission
from .masl import SIMULATION_PARAMETERS, RayleighChannel, create_env
from .masl import MobileUser as MASLMobileUser, Profile as MASLProfile
from simofld import masl
logger = getLogger(__name__)


class DQNAgent:
    def __init__(self, state_size, action_size, enable_dueling=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.8  # discount rate
        self.alpha = 0.9 # Q learning rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay_factor = 0.99
        self.learning_rate = 0.03
        if enable_dueling is None:
            enable_dueling = SIMULATION_PARAMETERS.get('ENABLE_DUELING', False)
        self.model = self._build_model(enable_dueling)
        self.target_model = self._build_model(enable_dueling)

    def _build_model(self, enable_dueling=False):
        if enable_dueling:
            input_shape = (self.state_size,)
            output_shape = (self.action_size,)
            X_input = keras.layers.Input(shape=input_shape)
            X = X_input

            # 'Dense' is the basic form of a neural network layer
            # Input Layer of state size(4) and Hidden Layer with 512 nodes
            X = keras.layers.Dense(12, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

            # Hidden layer with 256 nodes
            X = keras.layers.Dense(6, activation="relu", kernel_initializer='he_uniform')(X)
            
            # Hidden layer with 64 nodes
            X = keras.layers.Dense(3, activation="relu", kernel_initializer='he_uniform')(X)

            state_value = keras.layers.Dense(1, kernel_initializer='he_uniform')(X)
            state_value = keras.layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=output_shape)(state_value)

            action_advantage = keras.layers.Dense(self.action_size, kernel_initializer='he_uniform')(X)
            action_advantage = keras.layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=output_shape)(action_advantage)

            X = keras.layers.Add()([state_value, action_advantage])

            model = keras.Model(inputs = X_input, outputs=X)
            model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=["accuracy"])

            model.summary()
            return model
        else:
            # Neural Net for Deep-Q learning Model
            init = tf.keras.initializers.he_uniform()
            model = keras.Sequential()
            model.add(keras.layers.Dense(12, input_dim=self.state_size, activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(8, activation='relu', kernel_initializer=init))
            model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
            model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
            return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return py_random.randrange(self.action_size)
        act_values = self.model.predict(state[None, :])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = py_random.sample(self.memory, batch_size)

        current_states = np.array([e[0] for e in minibatch])
        current_q_list = self.model.predict(current_states)
        
        next_states = np.array([e[3] for e in minibatch])
        future_max_q_list = np.max(self.target_model.predict(next_states), axis=1)
        
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

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def copy_weights_to_target(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class MobileUser(MASLMobileUser):
    def __init__(self, channels: List[RayleighChannel], distance: Optional[Number]=None, active_probability: Optional[Number]=None, run_until: Optional[Number]=None, enable_dueling=False) -> None:
        super().__init__(channels=channels, distance=distance, active_probability=active_probability)
        self.payoff_logs = deque(maxlen=100)
        self._run_until = None
        self.dqn_agent = DQNAgent(state_size=len(self.channels) + 1, action_size=len(self.channels) + 1, enable_dueling=enable_dueling)

    def reward(self, choice_index):
        if choice_index == 0:
            payoff = self._Q()
        else:
            channel = self.channels[choice_index - 1]
            payoff = self._I(channel)
        
        r = 1 - 1e6 * payoff
        r = max(min(r, 10), -10)
        return r

    def log_payoffs(self):
        payoffs = [self._Q()] + [self._I(channel) for channel in self.channels]
        self.payoff_logs.append(payoffs)

    def get_state(self):
        recent_n = 8

        if self.payoff_logs:
            recent_n = min(len(self.payoff_logs), recent_n)
            weight = 1 
            state = np.zeros(len(self.channels) + 1)
            for i in range(recent_n):
                state += 1e6 * weight * np.array(self.payoff_logs[-(i+1)])
                weight *= 0.98
            state = state.clip(0, 50)
            return state
        else:
            return np.zeros(len(self.channels) + 1)


    async def main_loop(self):
        batch_size = 64

        last_transmission: Optional[Transmission] = None
        cloud_server = self._get_cloud_server()
        step_interval = self.get_current_env().g.step_interval
        self._choice_index = random.randint(0, len(self.channels) + 1)
        update_count = 0
        epsilon_decay_started = False
        while True:
            if self._run_until is not None and self.get_current_env().now > self._run_until:
                break

            state = self.get_state()
            self._choice_index = self.dqn_agent.act(state)
            await envs.wait_for_simul_tasks()
            if last_transmission:
                last_transmission.disconnect()
                last_transmission = None

            self.active = True if random.random() < self.active_probability else False
            if self.active:
                update_count += 1
                if self._choice_index == 0:
                    await self.perform_local_computation(step_interval)
                else:
                    channel = self.channels[self._choice_index - 1]
                    last_transmission = channel.connect(self, cloud_server)
                    await self.perform_cloud_computation(cloud_server, channel, step_interval)                
                reward = self.reward(self._choice_index)
                self.log_payoffs()
                new_state = self.get_state()
                action = self._choice_index
                self.dqn_agent.memorize(state, action, reward, new_state, False)                
                
                if len(self.dqn_agent.memory) > batch_size:
                    if update_count % 4 == 0:
                        epsilon_decay_started = True
                        self.dqn_agent.replay(batch_size)
                    if update_count > 30:
                        self.dqn_agent.copy_weights_to_target()
                        update_count = 0
            else:
                update_count += 1
                await envs.sleep(step_interval)

            if epsilon_decay_started:
                self.dqn_agent.decay_epsilon()

class CloudServer(masl.CloudServer):
    pass

class Profile(MASLProfile):
    def sample(self):
        temp_nodes = []
        for node in self.nodes:
            if node._run_until is None:
                temp_nodes.append(node)
            elif node._run_until > envs.get_current_env().now:
                temp_nodes.append(node)
        self.nodes = temp_nodes
        alpha = SIMULATION_PARAMETERS['PATH_LOSS_EXPONENT']
        self.channel_powers_0 = np.array([node._x**(-alpha) * node.transmit_power for node in self.nodes]) # channel powers divied by respective rayleigh fading factors
        print(f'now: {envs.get_current_env().now}')
        return super().sample()
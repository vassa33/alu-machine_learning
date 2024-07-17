#!/usr/bin/env python3
"""
    Script to train a DQN agent to play Atari's Breakout
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


def create_model(num_actions):
    """
        Creates the CNN model for the DQN agent
    """
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=(4, 84, 84)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


def main():
    """
        Main function to train the DQN agent
    """
    env = gym.make('BreakoutDeterministic-v4')
    num_actions = env.action_space.n

    model = create_model(num_actions)
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory,
                   nb_steps_warmup=50000, target_model_update=10000,
                   policy=policy)
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=5000000, visualize=False, verbose=2)

    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    main()

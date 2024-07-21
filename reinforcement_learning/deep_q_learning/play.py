#!/usr/bin/env python3
"""
    Script to play Atari's Breakout using a trained DQN agent
"""

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from rl.agents import DQNAgent
from rl.policy import GreedyQPolicy


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
        Main function to play Breakout using the trained DQN agent
    """
    env = gym.make('BreakoutDeterministic-v4')
    num_actions = env.action_space.n

    model = create_model(num_actions)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy)
    dqn.compile()
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

import gym
from time import sleep

import json

from actor_critic import ActorCritic

starting_state = np.asarray(
    json.loads(open('./humanoid_starting_state.json').read()))

ACTION_BOUND = 0.4


def get_reward(state):
    # State should be a numpy array.
    diff = starting_state - state
    reward = np.mean(np.multiply(diff, diff))
    return -1 * reward


class Runner(object):
    def __init__(self, env, GAMMA=0.9):
        self.env = env
        self.states_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor_critic = ActorCritic(self.states_dim, self.action_dim)

    def play_random_game(self, render=True):
        observation = env.reset()

        for t in range(1000):
            if render == True:
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print('Episode finished after {} timesteps'.format(t + 1))
                break

    def play_game_from_actor(self, render=True, add_to_buffer=True):
        env = self.env
        obs = env.reset()
        for t in range(1000):
            if render == True:
                env.render()
            action = self.actor_critic.get_actions(
                np.asarray([obs]))[0]  # I think zero.
            new_obs, reward, done, info = env.step(action)

            if done:
                print('Episode finished after {} timesteps'.format(t + 1))
                break

            if add_to_buffer:
                real_reward = get_reward(new_obs)
                self.actor_critic.add_to_replay_buffer(obs, action,
                                                       real_reward, new_obs)

            obs = new_obs

    def train_from_replay_buffer(self):
        losses = self.actor_critic.train_from_replay_buffer()
        return np.mean(losses)


fake_env = {
    'observation_space': {
        'shape': [100]
    },
    'action_space': {
        'shape': [100]
    }
}


def run():
    num_games = 10000
    trains_per_game = 10
    env = gym.make('Humanoid-v1')
    runner = Runner(env)
    for i in range(10000):
        should_render = (i % 5 == 0)
        runner.play_game_from_actor(render=should_render)
        for j in range(trains_per_game):
            avg_loss = runner.train_from_replay_buffer()
            if j == 0:
                print("Average loss for that batch: {}".format(avg_loss))
    print("completed... Exiting.")


if __name__ == '__main__':
    run()

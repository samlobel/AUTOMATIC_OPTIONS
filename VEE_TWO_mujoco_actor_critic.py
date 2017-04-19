import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

import gym
from time import sleep
import json

from actor_critic import ActorCritic
"""
What Next: I need to do a few things. First, I need to write a way to
get the mean and stddev of all the states. Should be easy.

The hard part is going to be incorporating that into the model. It's
most important in regards to the reward. Maybe for this one,
it should just be that we transform the starting-state by this value,
because that's the only goal.

Will that work? Yeah, I think so. Morph reward, and we should be alright.

For the other one, it'll be harder.


Is there a way that batch normalization can help? Let me think...

It's definitely a huge part of my problem. I think both parts are.

But, the problem with batch normalization is that we're trying to match states here.
I sort of did batch normalization, but it wasn't a good one. I think maybe I should
maintain a BIG moving average for Batch Norm, and normalize the starting state with
this moving guy. That's not a bad idea!

I do think maybe first I should clean this up, or make a commit, or something.

There are a few ways of doing this averaging. The easiest one will get slow REALLY
fast. But I'll do it anyways, because I really don't care about stuff like that.


"""

STARTING_STATE_LOC = './mujoco_data/humanoid_starting_state.json'
MEAN_STATE_LOC = './mujoco_data/mean_state.json'
STDDEV_STATE_LOC = './mujoco_data/stddev_state.json'
MIN_STATE_LOC = './mujoco_data/min_state.json'
SPREAD_STATE_LOC = './mujoco_data/spread_state.json'



with open(STARTING_STATE_LOC) as f:
    STARTING_STATE = np.asarray(json.loads(f.read()))

with open(MEAN_STATE_LOC) as f:
    MEAN_STATE = np.asarray(json.loads(f.read()))

with open(STDDEV_STATE_LOC) as f:
    STDDEV_STATE = np.asarray(json.loads(f.read()))

with open(MIN_STATE_LOC) as f:
    MIN_STATE = np.asarray(json.loads(f.read()))

with open(SPREAD_STATE_LOC) as f:
    SPREAD_STATE = np.asarray(json.loads(f.read()))


def shift_state(state):
    # return ((state - MEAN_STATE) / (STDDEV_STATE))
    return ((state - MIN_STATE) / SPREAD_STATE)


SHIFTED_GOAL = shift_state(STARTING_STATE)
print SHIFTED_GOAL

ACTION_BOUND = 0.4


def get_reward(state):
    # State should be a numpy array.
    # print(state)
    diff = SHIFTED_GOAL - state
    reward = np.mean(np.multiply(diff, diff))
    print reward
    return -1 * reward

# def get_reward(old_state, new_state):
#     diff = old_state - new_state
#     reward = np.mean(np.multiply(diff, diff))
#     return -1 * reward

class Runner(object):
    def __init__(self, env, GAMMA=0.5):
        self.env = env
        self.states_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor_critic = ActorCritic(
            self.states_dim, self.action_dim, lr=0.0000000001)
        self.all_observations = np.asarray([])

    def get_means_stddevs(self, num_games=100, min_std_dev=0.01):
        observations = []
        env = self.env
        for i in xrange(num_games):
            obs = env.reset()
            while True:
                observations.append(obs)
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    print('game {} done'.format(i))
                    break
        observations = np.asarray(observations)
        mean = np.mean(observations, axis=0)
        stddev = np.maximum(np.std(observations, axis=0), min_std_dev)
        return mean, stddev    

    def write_mean_stddev_to_file(self, num_games=100, min_std_dev=0.01):
        mean, stddev = self.get_means_stddevs(num_games, min_std_dev)
        with open('./mujoco_data/mean_state.json', 'w') as f:
            f.write(json.dumps(mean.tolist()))
        with open('./mujoco_data/stddev_state.json', 'w') as f:
            f.write(json.dumps(stddev.tolist()))
        print('written')

    def get_min_spread(self, num_games=100, min_spread=0.05):
        observations = []
        env = self.env
        for i in xrange(num_games):
            obs = env.reset()
            while True:
                observations.append(obs)
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    print('game {} done'.format(i))
                    break
        observations = np.asarray(observations)
        min_obs = observations.min(axis=0)
        max_obs = observations.max(axis=0)
        spread = np.maximum(max_obs - min_obs, min_spread)
        return min_obs, spread

    def write_min_spread_to_file(self, num_games=100, min_spread=0.05):
        min_obs, spread = self.get_min_spread(num_games, min_spread)
        print(min_obs)
        print(spread)
        print(min_obs.shape, spread.shape)
        with open('./mujoco_data/min_state.json', 'w') as f:
            f.write(json.dumps(min_obs.tolist()))
        with open('./mujoco_data/spread_state.json', 'w') as f:
            f.write(json.dumps(spread.tolist()))
        print('written')


    def play_random_game(self, render=True):
        env = self.env
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
                sleep(0.05)
            obs = np.asarray(obs)
            shifted_obs = shift_state(obs)
            action = self.actor_critic.get_actions(
                np.asarray([shifted_obs]))[0]  # I think zero.
            new_obs, reward, done, info = env.step(action)

            if done:
                print('Episode finished after {} timesteps'.format(t + 1))
                break

            if add_to_buffer:
                shifted_new_obs = shift_state(new_obs)
                # real_reward = get_reward(shifted_obs, shifted_new_obs)
                real_reward = get_reward(shifted_new_obs)
                self.actor_critic.add_to_replay_buffer(
                    shifted_obs, action, real_reward, shifted_new_obs)

            obs = new_obs

    def play_game_from_actor_with_random(self,
                                         render=True,
                                         add_to_buffer=True,
                                         prob_random=0.05):
        env = self.env
        obs = env.reset()
        for t in range(1000):
            if render == True:
                env.render()
                sleep(0.01)
            obs = np.asarray(obs)
            shifted_obs = shift_state(obs)

            action = self.actor_critic.get_actions(
                np.asarray([shifted_obs]))[0]  # I think zero.
            if not render:
                for i in range(len(action)):
                    if random.random() < prob_random:
                        action[i] = (random.random()*0.8)-0.4

            # random_move = random.random() < prob_random
            # if random_move and not render:
            #     print('Random move!')
            #     action = env.action_space.sample()
            # else:
            #     action = self.actor_critic.get_actions(
            #         np.asarray([shifted_obs]))[0]  # I think zero.
            new_obs, reward, done, info = env.step(action)

            if done:
                print obs, '\n'
                print new_obs, '\n'
                print shifted_obs, '\n'
                exit()
                if add_to_buffer:
                    real_reward = -0.10
                    self.actor_critic.add_to_replay_buffer(shifted_obs, action, real_reward, shifted_obs)
                print('Episode finished after {} timesteps'.format(t + 1))
                break

            if add_to_buffer:
                shifted_new_obs = shift_state(new_obs)
                # real_reward = get_reward(shifted_obs, shifted_new_obs)
                real_reward = get_reward(shifted_new_obs)
                self.actor_critic.add_to_replay_buffer(shifted_obs, action,
                                                       real_reward, new_obs)

            obs = new_obs

    def train_from_replay_buffer(self, should_print):
        losses = self.actor_critic.train_from_replay_buffer(should_print)
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
    for i in range(100000):
        should_render = (i % 10 == 0)
        runner.play_game_from_actor_with_random(
            render=should_render, prob_random=0.1)
        for j in range(trains_per_game):
            avg_loss = runner.train_from_replay_buffer((j == 0))
            if j == 0:
                print("Average loss for that batch: {}".format(avg_loss))
    print("completed... Exiting.")

def PLAY_RANDOM():
    num_games = 10000
    env = gym.make('Humanoid-v1')
    runner = Runner(env)
    for i in range(num_games):
        runner.play_random_game()


def create_averages():
    num_games = 100
    env = gym.make('Humanoid-v1')
    runner = Runner(env)
    # runner.write_mean_stddev_to_file(100, min_std_dev=0.5)
    runner.write_min_spread_to_file(500, min_spread=0.05)


if __name__ == '__main__':
    run()
    # PLAY_RANDOM()
    # create_averages()

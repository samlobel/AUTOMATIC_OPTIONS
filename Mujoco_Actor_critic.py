import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

import gym
from time import sleep
import json

from actor_critic import ActorCritic
from min_spread_holder import MinSpreadHolder
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

GOAL_STATE_LOC = './mujoco_data/intermediate_state.json'
with open(GOAL_STATE_LOC) as f:
    GOAL_STATE = np.asarray(json.loads(f.read()))

ACTION_BOUND = 0.4


class Runner(object):
    def __init__(self, env, GOAL_STATE, GAMMA=0.95, lr=0.001):
        self.env = env
        self.GOAL_STATE = GOAL_STATE
        self.states_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor_critic = ActorCritic(
            self.states_dim, self.action_dim, GAMMA=GAMMA, lr=lr)
        self.min_spread_holder = MinSpreadHolder(self.states_dim)

    def render_if_true(self, render):
        if render:
            self.env.render()

    def get_reward(self, state):
        shifted_goal_state = self.shift_observation(self.GOAL_STATE)
        diff = state - shifted_goal_state
        reward = -1 * np.mean(np.multiply(diff, diff))
        return reward

    def add_observed_batch(self, obs_batch):
        self.min_spread_holder.add_batch(obs_batch)

    def shift_observation(self, obs):
        return self.min_spread_holder.transform(obs)

    def play_random_game(self, render=True, add_to_all_observations=False):
        env = self.env
        observation = env.reset()
        games_observations = []

        for t in range(1000):
            games_observations.append(observation)
            self.render_if_true(render)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                if add_to_all_observations:
                    self.add_observed_batch(np.asarray(games_observations))
                print('Episode finished after {} timesteps'.format(t + 1))
                break

    def play_game_from_actor_with_random(self,
                                         render=True,
                                         add_to_buffer=True,
                                         prob_random=0.0):
        games_observations = []
        env = self.env
        obs = env.reset()
        games_observations = []
        for t in range(1000):
            self.render_if_true(render)
            obs = np.asarray(obs)
            games_observations.append(obs)
            shifted_obs = self.shift_observation(obs)

            action = self.actor_critic.get_actions(
                np.asarray([shifted_obs]))[0]  # I think zero.
            if not render and (random.random() < prob_random):
                action = env.action_space.sample()
            # if not render:
            #     for i in range(len(action)):
            #         if random.random() < prob_random:
            #             action[i] = (random.random() * 0.8) - 0.4

            new_obs, reward, done, info = env.step(action)
            shifted_new_obs = self.shift_observation(new_obs)
            if add_to_buffer:
                # real_reward = 0.0 if not done else -1.0
                real_reward = self.get_reward(
                    shifted_new_obs) if not done else -2.0
                self.actor_critic.add_to_replay_buffer(
                    shifted_obs, action, real_reward, shifted_new_obs)
            if done:
                self.add_observed_batch(np.asarray(games_observations))
                print('Episode finished after {} timesteps'.format(t + 1))
                break

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
    runner = Runner(env, GOAL_STATE, GAMMA=0.95, lr=0.00001)
    runner.play_random_game(render=False, add_to_all_observations=True)
    for i in range(100000):
        should_render = (i % 10 == 0)
        runner.play_game_from_actor_with_random(
            render=should_render,
            add_to_buffer=(not should_render),
            prob_random=0.1)
        if i > 20:
            for j in range(trains_per_game):
                avg_loss = runner.train_from_replay_buffer((j == 0))
                if j == 0:
                    print("Average loss for that batch: {}".format(avg_loss))
    print("completed... Exiting.")


def PLAY_RANDOM():
    num_games = 10000
    env = gym.make('Humanoid-v1')
    runner = Runner(env, GOAL_STATE)
    for i in range(num_games):
        runner.play_random_game()


def WRITE_RANDOM():
    env = gym.make('Humanoid-v1')
    runner = Runner(env, GOAL_STATE)
    obs = env.reset()
    for i in range(4):
        obs, _, _, _ = env.step(env.action_space.sample())
        obs = obs
    string_repr = json.dumps(obs.tolist())
    with open('./mujoco_data/intermediate_state.json', 'w') as f:
        f.write(string_repr)


if __name__ == '__main__':
    run()
    # WRITE_RANDOM()
    # PLAY_RANDOM()
    # create_averages()

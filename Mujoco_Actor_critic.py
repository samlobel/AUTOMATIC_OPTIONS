import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

import gym
from time import sleep


from ActorCritic import Actor, Critic


class ActorCriticEnv(object):
  def __init__(self, env):
    self.env = env
    print('obs space shape: {}'.format(self.env.observation_space.shape))
    print('action space shape: {}'.format(self.env.action_space.shape))
    self.states_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]
    print('states dim: {}\t\t actions dim: {}'.format(self.states_dim, self.action_dim))
    self.actor = Actor(self.states_dim, self.action_dim)
    self.critic = Critic(self.states_dim, self.action_dim)

  def play_random_game(self, render=True):
    observation = env.reset()
    for t in range(1000):
      if render == True:
        env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      if done:
        print('Episode finished after {} timesteps'.format(t+1))
        break



if __name__ == '__main__':
  env = gym.make('Humanoid-v1')
  ACE = ActorCriticEnv(env)
  for i in range(10):
    ACE.play_random_game()




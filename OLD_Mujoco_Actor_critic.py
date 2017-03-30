import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

import gym
from time import sleep

import json
from ActorCritic import Actor, Critic
import random
from replay_buffer import ReplayBuffer


starting_state = np.asarray(json.loads(open('./humanoid_starting_state.json').read()))


# What to do now? Well, I could train it each step, with
# a REALLY small step size. I sort of like that.


# For this one, as a test, I could just train it to stay at the start. 


def get_reward(state):
  # State should be a numpy array.
  diff = starting_state - state
  reward = np.mean(np.multiply(diff, diff))
  return -1*reward


def random_with_prob(prob):
  return (random.random() <= prob)





class ActorCriticEnv(object):
  def __init__(self, env, GAMMA=0.9):
    self.env = env
    print('obs space shape: {}'.format(self.env.observation_space.shape))
    print('action space shape: {}'.format(self.env.action_space.shape))
    self.states_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.shape[0]
    print('states dim: {}\t\t actions dim: {}'.format(self.states_dim, self.action_dim))
    self.actor = Actor(self.states_dim, self.action_dim, lr=0.0001)
    self.critic = Critic(self.states_dim, self.action_dim, lr=0.0001)
    self.GAMMA = GAMMA
    self.RANDOM_PROB = 0.025
    self.replay_buffer = ReplayBuffer(1280)

  def add_state_action_to_buffer(self, state, action, resulting_state, done):
    if done:
      predicted_q_val = np.asarray([[-25000]])
    else:
      best_new_action = self.actor.get_action(np.asarray([resulting_state]))
      predicted_next_q = self.critic.predict_q_val(np.asarray([resulting_state]), best_new_action)

      true_reward = get_reward(resulting_state)
    self.replay_buffer.add(state, action, true_reward, 0, resulting_state)
    # The 0 is for "t", which I don't understand the point of.
    return


  def train_from_state_action(self, state, action, resulting_state, done):
    if done:
      predicted_q_val = np.asarray([[-25000]])
    else:
      best_new_action = self.actor.get_action(np.asarray([resulting_state]))
      predicted_next_q = self.critic.predict_q_val(np.asarray([resulting_state]), best_new_action)

      true_reward = get_reward(resulting_state)

      predicted_q_val = true_reward + self.GAMMA*predicted_next_q


    wrapped_state = np.asarray([state])
    wrapped_action = np.asarray(action)
    # wrapped_q_goal = np.asarray([[predicted_q_val]])

    # print("STATE SHAPE: {}\t\tACTION SHAPE: {}\t\tREWARD SHAPE: {}".format(wrapped_state.shape, wrapped_action.shape, wrapped_true_reward.shape))

    inputs = [wrapped_state,wrapped_action,predicted_q_val]
    # print('created inputs. Calculating action grads.')
    action_grads = self.critic.get_action_grads(*inputs)
    # print('Optimizing critic q-val prediction.')    
    self.critic.optimize_q_val(*inputs)
    # print('training actor from state and grads')
    self.actor.train_from_batch(wrapped_state, action_grads)
    # print('all done training')

  # def train_from_replay_buffer(self, batch_size=64):
  #   s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)
  #   best_new_actions = self.actor.get_action(s2_batch)
  #   s2_predicted_q_vals = self.critic.predict_q_val(s2_batch, best_new_actions)


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

  def play_game_from_actor(self, render=True):
    observation = env.reset()
    for t in range(1000):
      if render == True:
        env.render()
      # action = env.action_space.sample()
      print(observation)
      action = self.actor.get_action(np.asarray([observation]))
      observation, reward, done, info = env.step(action)
      if done:
        print('Episode finished after {} timesteps'.format(t+1))
        break

  def train_actor_critic_to_stay_still(self, render=True):
    # My reward after each one is the difference between where you are
    # and where you started.
    true_rewards = []
    observation = env.reset()
    for t in range(1000):
      if render == True:
        env.render()

      true_rewards.append(get_reward(observation))
      if random_with_prob(self.RANDOM_PROB):
        action = np.asarray([env.action_space.sample()])
      else:
        action = self.actor.get_action(np.asarray([observation]))
      new_observation, reward, done, info = env.step(action)

      self.train_from_state_action(observation, action, new_observation, done)
      observation = new_observation

      if done:
        print('Episode finished after {} timesteps. Average reward: {}'.format(t+1, np.mean(np.asarray(true_rewards))))
        break




if __name__ == '__main__':
  env = gym.make('Humanoid-v1')
  ACE = ActorCriticEnv(env)
  for i in range(10000):
    # ACE.play_random_game()
    # ACE.play_game_from_actor()
    should_render = (i % 5 == 0)
    ACE.train_actor_critic_to_stay_still(should_render)




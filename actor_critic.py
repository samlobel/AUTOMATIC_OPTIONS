import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

from actor import Actor
from critic import Critic

from replay_buffer import ReplayBuffer

DEFAULT_LR = 0.01


def a_if_prob_else_b(a, b, eps):
    # Chooses b with prob eps, else a.
    if eps < 0 or eps > 1:
        raise Exception('bad prob: {}'.format(eps))
    return (b if random.random() < eps else a)


def random_string(N):
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(N))


class ActorCritic(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 final_activation=tf.nn.tanh,
                 action_bound=0.4,
                 training_batch_size=32,
                 GAMMA=0.95,
                 lr=0.001,
                 replay_buffer_size=1024):
        self.ID = random_string(10)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.final_activation = final_activation
        self.action_bound = action_bound
        self.GAMMA = GAMMA
        self.lr = lr
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.training_batch_size = training_batch_size
        with tf.variable_scope(self.ID) as scope:
            self.actor = Actor(self.state_dim, self.action_dim,
                               self.action_bound, self.lr,
                               self.final_activation)
            self.critic = Critic(self.state_dim, self.action_dim, self.lr)

    def add_to_replay_buffer(self, state, action, reward, resulting_state):
        self.replay_buffer.add(state, action, reward, resulting_state)

    def add_batch_to_replay_buffer(self, states, actions, rewards,
                                   resulting_states):
        for s, a, r, rs in zip(states, actions, rewards, resulting_states):
            self.replay_buffer.add(s, a, r, rs)

    def get_batch(self, training_batch_size=None):
        if not training_batch_size:
            training_batch_size = self.training_batch_size
        return self.replay_buffer.sample_batch(training_batch_size)

    def train_from_replay_buffer(self, should_print=False):
        # small trouble: if it's done, you don't want to run this thing on it.
        # I takes the new state, I predict an action, I predict that pair's q val,
        # I do: reward + GAMMA*next_q_val. I then do critic.optimize_q_val
        if not self.replay_buffer.size():
            print('buffer empty!')
            return 0
        states, actions, rewards, resulting_states = self.replay_buffer.sample_batch(
            self.training_batch_size)
        predicted_action = self.actor.get_actions(resulting_states)
        predicted_vals = self.critic.predict_q_val(resulting_states,
                                                   predicted_action)
        true_vals = rewards + (self.GAMMA * predicted_vals)
        # print(true_vals[4])
        losses = self.critic.optimize_q_val(states, actions, true_vals)
        grads = self.critic.get_action_grads(states, actions)
        self.actor.train_from_batch(states, grads)
        return losses
        if should_print:
            actual_q, out = self.critic.return_q_and_out(states, actions, true_vals)
            print('ACTUAL_Q: {}\n\n'.format(actual_q))
            print('OUT: {}'.format(out))
        return losses

    def get_actions(self, states):
        return self.actor.get_actions(states)

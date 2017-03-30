from actor_critic import ActorCritic
import tensorflow as tf
import prettytensor as pt
import numpy as np
"""
How about I think for a second before I start writing.
First, I don't want to extend actor_critic, I want to have an
ActorCritic inside of the object. That's because some of the
functionality is going to be a little different. 
To the actor, I'm going to pass in a (state+goal_state), and receive
an action. And then to the critic, I'm going to pass in a
the two states and the action, and return a value.
In other words, the state is now just the state plus the goal state.
So, I just double the state-bound. And when I pass things in, I want to
concatenate them first.
So, I guess it's not so bad.
I just need to add them both to the replay buffer.
It's too bad that there's not a good way for the thing to know what
the relationships are. I wish there was. 

Let's see if I can't figure out how to do this. Using super maybe?

I think this would be a LOT cleaner if we only had resulting_state
have the goal concatenated. I can deal with how, later.

"""


class StateController(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound=0.4,
                 training_batch_size=32,
                 GAMMA=0.95,
                 lr=0.001,
                 replay_buffer_size=1024):

        self.AC = ActorCritic(
            new_state_dim,
            action_dim,
            action_bound=action_bound,
            training_batch_size=training_batch_size,
            GAMMA=GAMMA,
            lr=lr,
            replay_buffer_size=replay_buffer_size)

    def get_reward(self, resulting_state, goal_state):
        return np.sum(((resulting_state - goal_state)**2), 1)

    def add_to_replay_buffer(self, state, goal_state, action, resulting_state):
        combined_state = np.concatenate(
            state, goal_state)  #combined is state plus goal
        reward = self.get_reward(resulting_state,
                                 goal_state)  # But reward is result - goal
        real_resulting_state = np.concatenate(resulting_state, goal_state)
        self.AC.add_to_replay_buffer(combined_state, action, reward,
                                     real_resulting_state)

    def add_batch_to_replay_buffer(self, states, goal_states, actions,
                                   resulting_states):
        for s, gs, a, rs in zip(states, goal_states, actions, rewards,
                                resulting_states):
            self.AC.add_to_replay_buffer(s, gs, a, rs)

    def train_from_replay_buffer(self):
        self.AC.train_from_replay_buffer()

    def get_actions(self, states, goal_states):
        combined_states = np.concatenate((states, goal_states), 1)
        return self.AC.get_actions(combined_states)


class GoalController(object):
    def __init__(self,
                 state_dim,
                 action_bound=1.0,
                 final_activation=tf.identity,
                 training_batch_size=32,
                 GAMMA=0.95,
                 lr=0.001,
                 replay_buffer_size=1024):

        self.AC = ActorCritic(
            state_dim,
            state_dim,
            final_activation=final_activation,
            action_bound=action_bound,
            training_batch_size=training_batch_size,
            GAMMA=GAMMA,
            lr=lr,
            replay_buffer_size=replay_buffer_size)

    def add_to_replay_buffer(self, state, goal_state, reward, resulting_state):
        # Here, reward means exactly what it sounds like it does...
        self.AC.add_to_replay_buffer(state, goal_state, reward,
                                     resulting_state)

    def add_batch_to_replay_buffer(self, states, goal_states, rewards,
                                   resulting_states):
        for s, gs, r, rs in zip(states, goal_states, rewards,
                                resulting_states):
            self.AC.add_to_replay_buffer(s, gs, r, rs)

    def train_from_replay_buffer(self):
        self.AC.train_from_replay_buffer()

    def get_goal_state(self, current_states):
        return self.AC.get_actions(current_states)



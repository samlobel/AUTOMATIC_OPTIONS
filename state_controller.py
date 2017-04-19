import numpy as np
from actor_critic import ActorCritic


class StateController(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound=0.4,
                 training_batch_size=32,
                 GAMMA=0.95,
                 lr=0.001,
                 replay_buffer_size=1024):

        self.state_dim = state_dim

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

    def get_visited_state(self):
        return self.replay_buffer.get_batch()[0][0]
        
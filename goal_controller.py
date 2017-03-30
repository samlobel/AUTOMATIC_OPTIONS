from actor_critic import ActorCritic


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

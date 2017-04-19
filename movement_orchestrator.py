"""
This is going to be pretty sexy. It's going to have a StateController and
a GoalController. It'll have methods to get either one's result. Also,
methods to train both. So, I don't know exactly how it should train itself.

Training: Should I have both things training at the same time? I really don't know.
One of the main reasons I don't want to do that is because the GoalController is going to
predict terrible states most of the time. They might be unobtainable, and throw everything
off!

So, I need a train_with_replay_buffer_goals method. It should use replay-buffer
visited-states in order to generate the new goals. It makes sense.

Can I change them every single time? Yeah I think so, Why the heck not?!
Although I do feel like maybe there's some value to consistency, especially in how it'll
help with exploration of a path to completion. So, I don't know...
I don't want to be pulled in opposite directions every single time, never getting
closer to the goal-state. That would be super annoying.

So, I could change it every time, or I could do a single goal for the entire round...
Or, I could change it every ten or so. I don't know when this should be
implemented. There should be a runner, but I think this isn't that. This is just
a Controller. 

Okay, I think: For the GoalController, we can change the goal every time, just like we
would do for a regular learner. For the StateController, we should change the goal less
frequently, so that it actually explores a little. So, for generating
things for the StateController replay buffer, we'll stay with the same goal every ten
tries.

"""
from controller import StateController, GoalController
import gym

def get_env_state_space_dim(env):
    return env.observation_space.shape[0]


def get_env_action_space_dim(env):
    return env.action_space.shape[0]


class MovementOrchestrator(object):
    def __init__(self,
                 env,
                 action_bound=0.4,
                 training_batch_size=32,
                 GAMMA=0.95,
                 lr=0.001,
                 replay_buffer_size=1024):
        self.env = env
        self.states_dim = states_dim = get_env_state_space_dim(env)
        self.action_dim = action_dim = get_env_action_space_dim(env)
        self.state_controller = StateController(self.states_dim, self.action_dim)
        self.goal_controller = GoalController(self.states_dim)

    def play_game_with_alternating_random_goals(self,
                                                render=True,
                                                add_to_buffer=True,
                                                switch_time=10):
        raise Exception("Need to check this method out when I'm awake")
        env = self.env
        obs = env.reset()
        goal = None
        for t in range(1000):
            if render == True:
                env.render()
            if t % switch_time == 0:
                goal = self.state_controller.get_random_visited_state()
            action = self.state_controller.get_actions(
                np.asarray([obs]), np.asarray([goal]))[0]

            new_obs, reward, done, info = env.step(action)

            if done:
                print('Episode finished after {} timesteps'.format(t + 1))
                break

            if add_to_buffer:
                self.state_controller.add_to_replay_buffer(obs, goal, action,
                                                           new_obs)

            obs = new_obs

    def play_game_with_calculated_goals(self, render=True, add_to_buffer=True):
        raise Exception("Need to check this method out when I'm awake")
        env = self.env
        obs = env.reset()
        for t in range(1000):
            if render == True:
                env.render()

            goal_arr = self.goal_controller.get_goal_state(np.asarray([obs]))
            goal = goal_arr[0]

            action = self.state_controller.get_actions(
                np.asarray([obs]), goal_arr)[0]

            new_obs, reward, done, info = env.step(action)

            if done:
                print('Episode finished after {} timesteps'.format(t + 1))
                break

            if add_to_buffer:
                self.goal_controller.add_to_replay_buffer(obs, goal, action,
                                                          new_obs)

            obs = new_obs

    def train_state_controller_from_replay_buffer(self):
        losses = self.state_controller.train_from_replay_buffer()
        return np.mean(losses)

    def train_goal_controller_from_replay_buffer(self):
        losses = self.goal_controller.train_from_replay_buffer()
        return np.mean(losses)


def run():
    num_games = 10000
    trains_per_game = 10
    env = gym.make('Humanoid-v1')
    orchestrator = MovementOrchestrator(env)
    for i in range(10000):
        should_render = (i % 5 == 0)
        orchestrator.play_game_with_alternating_random_goals(
            render=should_render)
        orchestrator.play_game_with_calculated_goals(render=should_render)
        for j in range(trains_per_game):
            avg_loss_state = orchestrator.train_state_controller_from_replay_buffer(
            )
            avg_loss_goal = orchestrator.train_goal_controller_from_replay_buffer(
            )
            if j == 0:
                print(
                    "Average loss:\t\t\tState Controller: {}\t\tGoal Controller: {}".
                    format(avg_loss_state, avg_loss_goal))

    print("completed... Exiting.")

if __name__ == '__main__':
    run()

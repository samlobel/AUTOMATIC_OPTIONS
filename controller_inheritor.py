# from actor_critic import ActorCritic
# import tensorflow as tf
# import prettytensor as pt
# """
# How about I think for a second before I start writing.
# First, I don't want to extend actor_critic, I want to have an
# ActorCritic inside of the object. That's because some of the
# functionality is going to be a little different. 
# To the actor, I'm going to pass in a (state+goal_state), and receive
# an action. And then to the critic, I'm going to pass in a
# the two states and the action, and return a value.
# In other words, the state is now just the state plus the goal state.
# So, I just double the state-bound. And when I pass things in, I want to
# concatenate them first.
# So, I guess it's not so bad.
# I just need to add them both to the replay buffer.
# It's too bad that there's not a good way for the thing to know what
# the relationships are. I wish there was. 

# Let's see if I can't figure out how to do this. Using super maybe?

# """


# class Controller(ActorCritic):
#     def __init__(self,
#                  state_dim,
#                  action_dim,
#                  action_bound=0.4,
#                  training_batch_size=32,
#                  GAMMA=0.95,
#                  lr=0.001,
#                  replay_buffer_size=1024):
#         new_state_dim = 2 * state_dim
#         super(
#             new_state_dim,
#             action_dim,
#             action_bound,
#             training_batch_size,
#             GAMMA,
#             lr,
#             replay_buffer_size)


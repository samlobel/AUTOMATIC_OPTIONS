import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

DEFAULT_LR = 0.01


MIN_ACTUATION = -0.4
MAX_ACTUATION = 0.4


def a_if_prob_else_b(a,b,eps):
  # Chooses b with prob eps, else a.
  if eps < 0 or eps > 1:
    raise Exception('bad prob: {}'.format(eps))
  return (b if random.random() < eps else a)

def random_string(N):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


class Actor(object):
  def __init__(self, state_dim, action_dim, action_bound=0.4, lr=0.01):
    self.ID = random_string(10)
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_bound = action_bound
    self.lr = lr
    with tf.variable_scope(self.ID) as scope:
      print('scope: {}'.format(scope))
      self.construct_model()
      self.construct_training()

  def construct_model(self):
    # All will be 4 layers, with elu, and a tanh at the end.
    with tf.variable_scope("Actor_model") as scope:
      print("SCOPE: {}".format(scope))
      h1_size = int((2*self.state_dim/3.0) + self.action_dim/3.0)
      h2_size = int((self.state_dim/3.0) + 2*self.action_dim/3.0)
      print('sizes descending:\t\t{}\t{}\t{}\t{}'.format(
        self.state_dim, h1_size, h2_size, self.action_dim))

      states_in = tf.placeholder(tf.float32, [None, self.state_dim])
      out = (pt.wrap(states_in)
        .fully_connected(h1_size, activation_fn=tf.nn.elu)
        .fully_connected(h2_size, activation_fn=tf.nn.elu)
        .fully_connected(self.action_dim, activation_fn=tf.nn.tanh)
      )
      out = out * self.action_bound
      self.in_batch = states_in
      self.out = out
      self.lr_var = tf.placeholder(tf.float32, [])
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.network_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=r".*Actor_model")

  def construct_training(self):
    with tf.variable_scope('Actor_training') as scope:
      self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
      self.actor_gradient = tf.gradients(self.out, self.network_parameters, -1*self.action_gradient)
      print([v.get_shape() for v in self.actor_gradient])
      zipped_grads_and_params = zip(self.actor_gradient, self.network_parameters)
      self.train = tf.train.AdamOptimizer(self.lr_var)\
        .apply_gradients(zipped_grads_and_params)
      print("Gradients applied for actor!")

  def initialize_session(self):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ID)
    init = tf.variables_initializer(variables)
    sess = tf.Session()
    sess.run(init)
    self.sess = sess

  def construct_feed_dict(self, update_dict={}):
    feed_dict = {
      self.lr_var : self.lr
    }
    feed_dict.update(update_dict)
    return feed_dict

  def get_action(self, states_in):
    feed_dict = self.construct_feed_dict({
      self.in_batch : states_in
    })
    return self.sess.run(self.out, feed_dict=feed_dict)

  def train_from_batch(self, states_in, grads_in):
    feed_dict = self.construct_feed_dict({
      self.in_batch : states_in,
      self.action_gradient : grads_in
    })
    self.sess.run(self.train, feed_dict=feed_dict) #I wish there was some metric to return but there isn't.



class Critic(object):
  def __init__(self, state_dim, action_dim, lr=0.01):
    self.ID = random_string(10)    
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.lr = lr
    with tf.variable_scope(self.ID):
      self.construct_model()
      self.construct_training()
    self.initialize_session()
  
  def construct_model(self):
    # I guess I should do the thing where I separate them by levels. Because they're on 
    # different footings, and I don't want the size of one to be way different from
    # the size of the other.
    # I'm not sure this model is good enough. 
    with tf.variable_scope("Critic_model"):
      self.states_in = tf.placeholder(tf.float32, [None, self.state_dim])
      self.actions_in = tf.placeholder(tf.float32, [None, self.action_dim])
      l1_states = pt.wrap(self.states_in).fully_connected(300, activation_fn=tf.nn.elu)
      l1_actions = pt.wrap(self.actions_in).fully_connected(300, activation_fn=tf.nn.elu)
      l1 = l1_states + l1_actions
      out = (pt.wrap(l1).fully_connected(100, activation_fn=tf.nn.elu)
              .fully_connected(1, activation_fn=tf.identity))
      self.out = out
    self.network_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=r".*Critic_model")

  def construct_training(self):
    with tf.variable_scope("Critic_training"):
      self.lr_var = tf.placeholder(tf.float32, [])
      self.actual_q_val = tf.placeholder(tf.float32, [None, 1]) #Reward + Gamma * q_val of next state.
      self.loss = tf.reduce_sum(tf.squared_difference(self.actual_q_val, self.out))
      self.train = tf.train.AdamOptimizer(self.lr_var).minimize(self.loss, var_list=self.network_parameters)
      self.action_grads = tf.gradients(self.out, self.actions_in) # It says to divide by the minibatch. Why is that? Whatever.
      print("gradients applied for critics!")

  def initialize_session(self):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ID)
    init = tf.variables_initializer(variables)
    sess = tf.Session()
    sess.run(init)
    self.sess = sess

  def construct_feed_dict(self, update_dict={}):
    feed_dict = {
      self.lr_var : self.lr
    }

    feed_dict.update(update_dict)
    return feed_dict


  def predict_q_val(self, states_in, actions_in):
    feed_dict = self.construct_feed_dict({
      self.states_in : states_in,
      self.actions_in : actions_in
    })
    return self.sess.run(self.out, feed_dict=feed_dict)

  def optimize_q_val(self, states_in, actions_in, actual_q_val):
    feed_dict = self.construct_feed_dict({
      self.states_in : states_in,
      self.actions_in : actions_in,
      self.actual_q_val : actual_q_val
    })
    return self.sess.run([self.loss, self.train], feed_dict=feed_dict)[0]

  def get_action_grads(self, states_in, actions_in, actual_q_val):
    feed_dict = self.construct_feed_dict({
      self.states_in : states_in,
      self.actions_in : actions_in,
      self.actual_q_val : actual_q_val
    })
    return self.sess.run(self.action_grads, feed_dict=feed_dict)

























# def Critic(object):
#   def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
#     pass

#   def construct_model(self):
#     pass


    

# class ActorCritic(object):
#   def __init__(in_size, out_size, energy_function, lr=0.01):
#     # This feels exhausting. I can do it, but there's a lot of
#     # typing and maybe not so much reward at the end. Because
#     # it's an established algorithm I think. Intrinsic curiosity
#     # already works well I think. The whole options thing though,
#     # that could be interesting. So it depends if I want to take it
#     # that far. 

#     pass


# # class DQN(object):
# #   """
# #   Takes in a state, outputs an action. Not going to be totally true to the 
# #   original implementation, but such is life.

# #   I think I'll give it the input and output space, and do something silly
# #   in between. Like have a four-layer network, and have the middle two 
# #   be interpolations. I like that enough to run with it.

# #   I need it to have actions between -1 and 1 for everything
# #   (actually it's -0.4 and 0.4).
# #   How about something simple like tf.nn.tanh(in) * 0.5? Why not.
# #   How about just tf.nn.tanh

# #   God dammit, I now remember that DQNs are a huge pain in the behind.

# #   Shit, I just remembered this is continuous control. That means that best
# #   move isn't really a thing anymore. On the bright side, it makes it a little
# #   easier to do your updates.

# #   I think in make_move, it should take in an environment, make a move,
# #   and return all the things that making a move return, as well 

# #   God dammit, there's another problem. I didn't think this through at all.

# #   There's no value for a specific output the way there would be for a regular
# #   DQN. That makes it really easy. Instead, I need to do some sort of
# #   value network thing.

# #   BUT, you can't really properly optimize based on the value network, because
# #   you don't have access to the value network deterministically.
# #   Unless, maybe you do! Because it's all tensorflow...

# #   God dammit. This is sort of tough, and definitely annoying. Maybe
# #   I should read a little about continuous control using NNs.

# #   Okay, one way to do it is this:
# #   You have an actor and a critic.
# #   The critic tries to mape STATES+ACTIONS into an expected value.
# #   This can be trained by seeing what actually happens.
# #   You can make random moves, and train the critic on them.
  
# #   The actor maps states to actions.




# #   http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html  is a great
# #   description of what I need to do. Looks like DQN isn't gonna work so well.


# #   Critic takes in states and actions, returns single value function.

# #   Actor takes in the gradients from the critic (only the ones relavent to the action)
# #   That's the path to follow to maximize/minimize the value, given the state.
# #   Then, the actor follows that path.

# #   You have to divide it up, because you want to sometimes try random actions, and
# #   get their value. That's because the actor is always on-policy, but the critic is not.
# #   Pretty interesting...

# #   Call "create_actor", then "create_critic", then call "create_critic_derivs", then "create_actor_derivs."





# #   """
# #   def __init__(in_size, out_size, energy_function, lr=0.01):
# #     self.in_size = in_size
# #     self.out_size = out_size
# #     self.lr = lr
# #     self.energy_function = energy_function
# #     self.construct_model()

# #   def construct_model(self):
# #     # All will be 4 layers, with elu, and a tanh at the end.
# #     h1_size = int((2*self.in_size/3.0) + self.out_size/3.0)
# #     h2_size = int((self.in_size/3.0) + 2*self.out_size/3.0)
# #     print('sizes descending:\t\t{}\t{}\t{}\t{}'.format(
# #       self.in_size, h1_size, h2_size, self.out_size))

# #     states_in = tf.placeholder(tf.float32, [None, self.in_size])
# #     out = (pt.wrap(states_in)
# #       .fully_connected(h1_size, activation_fn=tf.nn.elu)
# #       .fully_connected(h2_size, activation_fn=tf.nn.elu)
# #       .fully_connected(self.out_size, activation_fn=tf.nn.tanh)
# #     )
# #     out = out * 0.4
# #     self.in_batch = states_in
# #     self.out = out
# #     self.lr_var = tf.placeholder(tf.float32, [])

# #   def construct_loss(self):
# #     pass

# #   def construct_training(self):
# #     pass

# #   def initialize_session(self):
# #     pass

# #   def construct_feed_dict(self, update_dict={}):
# #     feed_dict = {
# #       self.lr_var : self.lr
# #     }

# #     feed_dict.update(update_dict)
# #     return feed_dict
  
# #   def calculate_best_moves(self, in_batch):
# #     feed_dict = self.construct_feed_dict({
# #       self.in_batch : in_batch
# #     })
# #     move_probs = self.sess.run(self.out, feed_dict=feed_dict)
# #     return move_probs

# #   def get_random_move(self):
# #     # returns a single random move
# #     return np.asarray([(random.random()*0.8 - 0.4) for _ in range(self.out_size)])

# #   def choose_moves(self, in_batch, eps=0.1):
# #     move_probs = self.get_move_probabilities(in_batch)
# #     move_or_random = np.asarray([a_if_prob_else_b(move, self.get_random_move(), eps) for move in move_probs])
# #     return move_or_random

# #   def train(self, in_batch):
# #     pass

# #   def get_variables(self, ):
# #     pass

# #   def set_learning_rate(new_lr):
# #     pass

import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random

DEFAULT_LR = 0.01

def random_string(N):
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

class AutoEncoder(object):
  """
  For a 100,10,5,10,100, you just pass in 100, 10, 5.

  Not sure if I'm doing this right...
  """
  def __init__(self, in_size, layer_nums=None, lr=0.01, downsample=2):
    self.ID = random_string(10)
    self.in_size = in_size
    self.lr = lr
    self.downsample=downsample
    # For now, ignores layer_nums, except for maybe in_size.
    # I'll just divide by two twice, and then multiply by two twice.
    with tf.variable_scope(self.ID):
      self.construct_model()
      self.construct_energy()
      self.construct_training()
    self.initialize_session()
    print('everything intialized.\t\tID: {}'.format(self.ID))

  def construct_model(self):
    # I want to make sure there's compression.
    in_size = self.in_size

    l2_size = in_size // self.downsample
    l3_size = in_size // (self.downsample**2)
    if l3_size < 1:
      raise Exception('pick a bigger in_size, drangus!')

    print('layer sizes:\t\t{}\t{}\t{}'.format(in_size, l2_size, l3_size))

    states_in = tf.placeholder(tf.float32, [None, in_size])
    out = (pt.wrap(states_in)
      .fully_connected(l2_size, activation_fn=tf.nn.elu)
      .fully_connected(l3_size, activation_fn=tf.nn.elu)
      .fully_connected(l2_size, activation_fn=tf.nn.elu)
      .fully_connected(in_size, activation_fn=tf.identity)
    ) #The last part is a little shitty. But so it goes.

    self.in_batch = states_in
    self.out = out
    self.lr_var = tf.placeholder(tf.float32, [])

  def construct_energy(self):
    self.energy_diff = tf.reduce_mean(tf.squared_difference(self.out, self.in_batch), reduction_indices=[1])
    self.net_energy_diff = tf.reduce_sum(self.energy_diff)


  def construct_training(self):
    # training_variables = tf.get_collection(tf.GraphKeys.TRAINING_VARIABLES, scope='my_scope')
    loss = self.net_energy_diff
    # training_op = tf.train.MomentumOptimizer(self.lr_var, 0.1).minimize(loss)
    training_op = tf.train.GradientDescentOptimizer(self.lr_var).minimize(loss)
    self.training_op = training_op

  def initialize_session(self):
    variables =  self.get_variables()
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

  def get_reconstruction(self, in_batch):
    feed_dict = self.construct_feed_dict({
      self.in_batch : in_batch
    })
    _out = self.sess.run(self.out, feed_dict=feed_dict)
    return _out

  def get_energies(self, in_batch):
    feed_dict = self.construct_feed_dict({
      self.in_batch : in_batch
    })
    _energy_diff = self.sess.run(self.energy_diff, feed_dict=feed_dict)
    return _energy_diff

  def train(self, in_batch):
    feed_dict = self.construct_feed_dict({
      self.in_batch : in_batch
    })
    energy, _ = self.sess.run([self.net_energy_diff, self.training_op], 
      feed_dict=feed_dict
    )
    return energy

  def get_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ID)

  def set_learning_rate(self, new_lr):
    self.lr = new_lr


def construct_random_input(shape):
  return np.random.rand(*shape)

def TEST():
  IN = 100
  a = AutoEncoder(in_size=IN)
  for i in range(1000):
    rand_in = construct_random_input((10,IN))
    en = a.train(rand_in)
    print(np.mean(en))
    recon = a.get_reconstruction(rand_in)
    # print("recon: {}".format(recon))
    diff = recon - rand_in
  b = AutoEncoder(in_size=IN)
  for i in range(1000):
    rand_in = construct_random_input((10,IN))
    en = b.train(rand_in)
    print(np.mean(en))
    recon = b.get_reconstruction(rand_in)
    # print("recon: {}".format(recon))
    diff = recon - rand_in
    # if i % 100 == 0:
    #   print('\n\n\nDIFF: {}\n\n\n'.format(diff))

if __name__ == '__main__':
  TEST()

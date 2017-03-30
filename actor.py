import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random


def random_string(N):
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(N))


class Actor(object):
    def __init__(self, state_dim, action_dim, action_bound=0.4, lr=0.01, final_activation=tf.nn.tanh):
        self.ID = random_string(10)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.final_activation = final_activation
        self.lr = lr
        with tf.variable_scope(self.ID) as scope:
            print('scope: {}'.format(scope))
            self.construct_model()
            self.construct_training()
            self.initialize_session()

    def construct_model(self):
        # All will be 4 layers, with elu, and a tanh at the end.
        with tf.variable_scope("Actor_model") as scope:
            print("SCOPE: {}".format(scope))
            h1_size = int((2 * self.state_dim / 3.0) + self.action_dim / 3.0)
            h2_size = int((self.state_dim / 3.0) + 2 * self.action_dim / 3.0)
            print('sizes descending:\t\t{}\t{}\t{}\t{}'.format(
                self.state_dim, h1_size, h2_size, self.action_dim))

            states_in = tf.placeholder(tf.float32, [None, self.state_dim])
            out = (pt.wrap(states_in)
                   .fully_connected(h1_size, activation_fn=tf.nn.elu)
                   .fully_connected(h2_size, activation_fn=tf.nn.elu)
                   .fully_connected(self.action_dim, activation_fn=self.final_activation))
            out = out * self.action_bound
            self.in_batch = states_in
            self.out = out
            self.lr_var = tf.placeholder(tf.float32, [])
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_parameters = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=r".*Actor_model")

    def construct_training(self):
        with tf.variable_scope('Actor_training') as scope:
            self.action_gradient = tf.placeholder(tf.float32,
                                                  [None, self.action_dim])
            self.actor_gradient = tf.gradients(
                self.out, self.network_parameters, -1 * self.action_gradient)
            zipped_grads_and_params = zip(self.actor_gradient,
                                          self.network_parameters)
            self.train = tf.train.AdamOptimizer(self.lr_var)\
              .apply_gradients(zipped_grads_and_params)
            print("Gradients applied for actor!")

    def initialize_session(self):
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ID)
        init = tf.variables_initializer(variables)
        sess = tf.Session()
        sess.run(init)
        self.sess = sess

    def construct_feed_dict(self, update_dict={}):
        feed_dict = {self.lr_var: self.lr}
        feed_dict.update(update_dict)
        return feed_dict

    def get_actions(self, states_in):
        feed_dict = self.construct_feed_dict({self.in_batch: states_in})
        return self.sess.run(self.out, feed_dict=feed_dict)

    def train_from_batch(self, states_in, grads_in):
        feed_dict = self.construct_feed_dict({
            self.in_batch: states_in,
            self.action_gradient: grads_in
        })
        self.sess.run(
            self.train, feed_dict=feed_dict
        )  #I wish there was some metric to return but there isn't.

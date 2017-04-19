import numpy as np
import tensorflow as tf
import prettytensor as pt
import string
import random


def random_string(N):
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(N))


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
            self.actions_in = tf.placeholder(tf.float32,
                                             [None, self.action_dim])
            l1_states = pt.wrap(self.states_in).fully_connected(
                300, activation_fn=tf.identity)
            l1_actions = pt.wrap(self.actions_in).fully_connected(
                300, activation_fn=tf.identity)
            l1 = tf.nn.elu(l1_states + l1_actions)
            # l1 = tf.concat([l1_states, l1_actions], 1)
            print('{} should be ?, 300'.format(l1.get_shape()))
            out = (pt.wrap(l1).fully_connected(100, activation_fn=tf.nn.elu)
                   .fully_connected(1, activation_fn=tf.identity)
                   .fully_connected(1, activation_fn=tf.identity)
                   .flatten(preserve_batch=False))
            print("out shape: {}".format(out.get_shape()))
            self.out = out
        self.network_parameters = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=r".*Critic_model.*")

    def construct_training(self):
        with tf.variable_scope("Critic_training"):
            self.lr_var = tf.placeholder(tf.float32, [])
            self.actual_q_val = tf.placeholder(
                tf.float32, [None])  #Reward + Gamma * q_val of next state.
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.actual_q_val, self.out))

            self.train = tf.train.AdamOptimizer(self.lr_var).minimize(
                self.loss, var_list=self.network_parameters)

            self.action_grads = tf.gradients(
                self.out, self.actions_in
            )[0]
            print("gradients applied for critics!")

    def initialize_session(self):
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=r'.*' + self.ID + '.*')
        init = tf.variables_initializer(variables)
        sess = tf.Session()
        sess.run(init)
        self.sess = sess

    def construct_feed_dict(self, update_dict={}):
        feed_dict = {self.lr_var: self.lr}
        feed_dict.update(update_dict)
        return feed_dict

    def predict_q_val(self, states_in, actions_in):
        feed_dict = self.construct_feed_dict({
            self.states_in: states_in,
            self.actions_in: actions_in
        })
        return self.sess.run(self.out, feed_dict=feed_dict)

    def optimize_q_val(self, states_in, actions_in, actual_q_val):
        feed_dict = self.construct_feed_dict({
            self.states_in: states_in,
            self.actions_in: actions_in,
            self.actual_q_val: actual_q_val
        })
        return self.sess.run([self.loss, self.train], feed_dict=feed_dict)[0]

    def return_q_and_out(self, states_in, actions_in, actual_q_val):
        feed_dict = self.construct_feed_dict({
            self.states_in: states_in,
            self.actions_in: actions_in,
            self.actual_q_val: actual_q_val
        })
        return self.sess.run([self.actual_q_val, self.out], feed_dict=feed_dict)

    def get_action_grads(self, states_in, actions_in):
        feed_dict = self.construct_feed_dict({
            self.states_in: states_in,
            self.actions_in: actions_in
        })
        return self.sess.run(self.action_grads, feed_dict=feed_dict)

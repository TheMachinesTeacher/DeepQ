#!/usr/bin/env python3 
import gym
from gym import wrappers
import tensorflow as tf
from Model import *

class ModelRun(Process):

    game = ""
    numActions = 0
    gamma = tf.constant(.9, dtype=tf.float32, shape=(1,))
    rHat = tf.placeholder(tf.float32, shape=(1,))

    def __init__(self, game, numActions):
        super(ModelRun, self).__init__()
        self.game = game
        self.numActions = numActions

    def squaredError(q, qHat):
        qResp = tf.reduce_max(q)
        qHatResp = tf.reduce_max(qHat)
        y = tf.add(rHat, tf.multiply(gamma, qHatResp))
        diff = tf.subtract(y, qResp)
        return tf.multiply(diff, diff)

    def run(self):
        done = False
        env = gym.make(game)
        env = wrappers.Monitor(env, "/tmp/gym-results")
        with tf.Session() as sess:
            s = tf.placeholder(tf.float32, shape=()) #TODO figure this out
            sPrime = tf.placeholder(tf.float32, shape=()) #TODO figure this out
            q = makeModel(x, numActions)
            qHat = makeModel(sPrime, numActions)
            with tf.name_scope('loss'):
                loss = squaredError(q, qHat) 
            while not done:
                env.render()
                observation, reward, done, info = env.step(action)
            env.reset()
            env.close()

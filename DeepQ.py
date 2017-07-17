#!/usr/bin/env python3 
import gym
from gym import wrappers
import tensorflow as tf
from Model import *

class ModelRun(Process):

    game = ""
    numActions = 0
    gamma = None
    rHat = None
    s = None
    sPrime = None
    q = None
    qWeights = None
    qBiases = None
    qHat = None
    qHatWeights = None
    qHatBiases = None
    rng = random.SystemRandom()
    transistionDB = []

    def __init__(self, game, numActions):
        super(ModelRun, self).__init__()
        self.game = game
        self.numActions = numActions
    	self.gamma = tf.constant(.9, dtype=tf.float32, shape=(1,))
    	self.rHat = tf.placeholder(tf.float32, shape=(1,))
    	self.s = tf.placeholder(tf.float32, shape=()) #TODO figure this out
    	self.sHatPrime = tf.placeholder(tf.float32, shape=()) #TODO figure this out
        self.q, self.qWeights, self.qBiases = makeModel(self.s, numActions)
        self.qHat, self.qHatWeights, self.qHatBiases = makeModel(self.sHatPrime, numActions)
    	self.rng = random.SystemRandom()

    def getTransistion(o): # TODO feed s into q when running it

    def squaredError(o):
    	qResp = tf.argmax(self.q)
    	action = self.sess.run(actionGraph, feed_dict={self.s:o})
    	oPrime, reward, done, info = env.step(action)
    	transistion = (o, action, reward, oPrime)
    	transistionDB.append(transistion)
    	tranHat = transistionDB[self.rng.randint(0, len(transistionDB)-1)]
        qHatResp = tf.reduce_max(qHat)
        hatAction = self.sess.run(qHatResp, feed_dict={self.sHatPrime:tranHat[3]})


        y = tf.add(self.rHat, tf.multiply(self.gamma, qHatResp)) # TODO self.rHat needs to be fed
        diff = tf.subtract(y, qResp)
        return tf.multiply(diff, diff)

    def run(self):
        done = False
        env = gym.make(game)
        env = wrappers.Monitor(env, "/tmp/gym-results")
        with tf.Session() as sess:
            with tf.name_scope('loss'):
                loss = self.squaredError(q, qHat) 
            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
            o, reward, done, info = env.step(0)
            oPrime, reward, done, info = env.step(0)
            while not done:
                #env.render()
                #train_step.run(feed_dict={s:o, sPrime:oPrime, rHat:reward})
            env.reset()
            env.close()

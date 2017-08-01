#!/usr/bin/env python3
import gym
from gym import wrappers
import cv2
from DeepQ import *
import numpy as np 
from multiprocessing import Process

class MachinePlay(Process):

	def __init__(self, game):
	    super(MachinePlay, self).__init__()
	    self.game = game

	# preprocess raw image to 80*80 gray image
	def preprocess(self, observation):
	    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
	    observation = observation[26:110,:]
	    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	    return np.reshape(observation,(84,84,1))

	def encodeOneHot(self, num, length):
	    retval = [0]*length;
	    retval[num] = 1
	    return retval

	def decodeOneHot(self, encoding):
	    arr = np.array(encoding)
	    return arr[np.argmax(arr)]

	def run(self):
	    done = False
	    env = gym.make(self.game)
	    env = wrappers.Monitor(env, "/tmp/gym-results", force=True)
	    sess = tf.Session()
	    actions = env.action_space.n
	    agent = DQN(actions)
	    observation, reward, done, info = env.step(0) 
	    observation = self.preprocess(observation)
	    agent.setInitState(observation)
	    while not done:
                env.render()
                action = self.decodeOneHot(agent.getAction())
                observation, reward, done, info = env.step(action)
                observation = self.preprocess(observation)
                agent.setPerception(observation,action,reward,done)
	    env.close()

if __name__ == '__main__':
	player = MachinePlay('CartPole-v0')
	player.start()

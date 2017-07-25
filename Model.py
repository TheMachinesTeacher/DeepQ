#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from TensorboardUtilities import *
import sys

def relu(layer, name=""):
    with tf.name_scope(name+'_activations'):
        activations = tf.nn.relu(layer)
        tf.summary.histogram(name+'_activations_histogram', activations)
        return activations

def convLayer(din, filtSize, strides, in_channels, out_channels, name=""):
    with tf.name_scope(name):
        with tf.name_scope(name+'_weights_scope'):
            w = tf.Variable(tf.truncated_normal([filtSize, filtSize, in_channels, out_channels], stddev=.1), name=name+'_weights')
            variable_summaries(w)
        with tf.name_scope(name+'_biases_scope'):
            b = tf.Variable(tf.zeros([out_channels]), name=name+'_biases')
            variable_summaries(b)
        with tf.name_scope(name+'_convlution'):
            convOut = tf.add(tf.nn.conv2d(din, w, strides=[1, strides, strides, 1], padding='SAME', name=name), b)
            tf.summary.histogram(name+'_convOut', convOut)
        activations = relu(convOut, name)
        return activations, w, b

def fcLayer(din, in_channels, out_channels, name=""):
    with tf.name_scope(name):
        with tf.name_scope(name+'_weights_scope'):
            w = tf.Variable(tf.truncated_normal([in_channels, out_channels], stddev=.1), name=name+'_weights')
            variable_summaries(w)
        with tf.name_scope(name+'_biases_scope'):
            b = tf.Variable(tf.zeros([out_channels]), name=name+'_biases')
            variable_summaries(b)
        with tf.name_scope(name+'_innerProduct'):
            fc = tf.add(tf.matmul(din, w), b)
            tf.summary.histogram(name+'_fcOut', fc)
        activations = relu(fc, name)
        return activations, w, b

'''
def softmaxLayer(din, in_channels, out_channels, name=""):
    with tf.name_scope('SofmaxLayer'+name):
        w = tf.Variable(tf.truncated_normal([in_channels, out_channels], stddev=.1, name='softmax_weights'+name, dtype=tf.float32))
        variable_summaries(w)
        b = tf.Variable(tf.truncated_normal([out_channels], stddev=.1, name='softmax_biaes'+name))
        variable_summaries(b)
        result = tf.matmul(din[:,0,:], w)+b
        tf.summary.histogram('SoftmaxLayer'+name, result)
        return result, w, b
'''

# numActions is the number of actions the agent can take in this game
# numFramesPerInput is also known as m
def DQNbyDeepMind(numActions, numFramesPerInput=4):
    x = tf.placeholder("float",[None,84,84,4])
    convFiltSizes = [8, 4, 3]
    convFiltStrides = [4, 2, 1]
    convFilts = [numFramesPerInput, 32, 64, 64]
    fcUnits = [convFilts[-1], 512, 512]
    
    layers = [x]
    weights = [tf.Variable(0)]
    biases = [tf.Variable(0)]
    for i in range(len(convFiltSizes)):
        name = 'conv'+str(i)
        l, w, b = convLayer(layers[-1], convFiltSizes[i], convFiltStrides[i], convFilts[i], convFilts[i+1], name);
        layers.append(l)
        weights.append(w)
        biases.append(b)
    for i in range(len(fcUnits)-1):
        name = 'fc'+str(i)
        l, w, b = fcLayer(layers[-1], fcUnits[i], fcUnits[i+1], name)
        layers.append(l)
        weights.append(w)
        biases.append(b)
    return layers, weights, biases

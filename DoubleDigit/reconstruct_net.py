from __future__ import division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf


def reconstruct(n_hidden1,n_hiden2,n_output):
	
	hidden1 = tf.layers.dense(decoder_input, n_hidden1,
							activation=tf.nn.relu,
							name="hidden1")
	hidden2 = tf.layers.dense(decoder_input, n_hidden2,
							activation=tf.nn.relu,
							name="hidden2")
	decoder_output = tf.layers.dense(hidden2,n_output,
							activation=tf.nn.sigmoid,
							name="decoder_output")






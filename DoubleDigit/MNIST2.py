from DataGenerator import generate_data
import numpy as np
from keras.datasets import mnist
import tensorflow as tf




class Data(object):

	def __init__(self, num_imgs, batch_size):
		raw_data = generate_data(num_imgs)
		self.batch_size = batch_size
		self.cooked_data = raw_data.batch(batch_size)
		self.iterator = self.cooked_data.make_one_shot_iterator()
		
	def next_batch():
		return self.iterator.get_next()



		
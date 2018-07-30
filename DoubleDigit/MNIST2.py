from DataGenerator import generate_data_np
import numpy as np
from keras.datasets import mnist
import tensorflow as tf




class Data_tf(object):

	def __init__(self, num_imgs, batch_size):
		# gets images together
		raw_data = generate_data_tf(num_imgs)
		self.batch_size = batch_size
		# batches images
		self.cooked_data = raw_data.batch(batch_size)
		# makes the batched images iterable
		self.iterator = self.cooked_data.make_one_shot_iterator()
		
	def next_batch(self):
		batch = self.iterator.get_next()
		batch = np.array(batch)
		return batch

class Data_np(object):

	def __init__(self,num_imgs,batch_size):
		self.data = generate_data_np(num_imgs)
		# keeps track of where we are in data set so we know what elements
		# to give when prompted
		self.data_set_marker = 0
		self.batch_size = batch_size

	def next_batch(self):
		front = self.data_set_marker
		tail = self.data_set_marker + self.batch_size
		self.data_set_marker = tail
		return (self.data[0][front:tail], self.data[1][front:tail])





		
from data_sew import ImageStitcher
import numpy as np
from keras.datasets import mnist
import tensorflow as tf

def generate_data_tf(num_imgs):
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_stitches = ImageStitcher(40, train_images, train_labels, overlap_range=(-17, 0))
	train_stitches.overlap_images(num_imgs = num_imgs)

	imgs = train_stitches.stitched_imgs
	labels = train_stitches.stitched_labels
	
	dataset = tf.data.Dataset.from_tensor_slices((imgs,labels))

	return dataset

def generate_data_np(num_imgs):
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_stitches = ImageStitcher(40, train_images, train_labels, overlap_range=(-17, 0),tuple_=True)
	train_stitches.overlap_images(num_imgs = num_imgs)

	return (train_stitches.stitched_imgs, train_stitches.stitched_labels)



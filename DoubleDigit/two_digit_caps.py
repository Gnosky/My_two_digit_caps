from __future__ import division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf 
### ---------
### ALERT!ALERT!
###
### This data_sew is a slightly altered version of the one that 
### Is in GitHub
### For details, see Grant
###
### ALERT!ALERT!
### ---------
from keras.datasets import mnist
from MNIST2 import Data_np

batch_size = 50
train_length = 10000
test_length = 5000
valid_length = 500

train = Data_np(train_length, batch_size)
test = Data_np(test_length,batch_size)
validation = Data_np(valid_length, batch_size)

np.random.seed(42)
tf.set_random_seed(42)


# Creates a placeholder for 28x56 images with one greyscale channel
X = tf.placeholder(shape=[None,28,56,1], dtype=tf.float32, name="X")

# The first capsule layer will have 32 * 6 * 6 capsules of 8 dimensions each
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 20
caps1_n_dims = 8

# -----First two Convolutions-----
conv1_params = {
	"filters":256,
	"kernel_size": 9,
	"strides": 1,
	"padding": "valid",
	"activation": tf.nn.relu,
}

conv2_params = {
	"filters":caps1_n_maps * caps1_n_dims,
	"kernel_size": 9,
	"strides": 2,
	"padding": "valid",
	"activation": tf.nn.relu,
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)



# -----Forming data into Primary Capsule Layer-----
# Note the epsilon arg. We add it to squared_norm so that we never divide by zero
caps1_raw = tf.reshape(conv2, [-1,caps1_n_caps, caps1_n_dims],
								name="caps1_raw")

def squash(s, axis=-1,epsilon=1e-7,name=None):
	with tf.name_scope(name,default_name="squash"):
		squared_norm = tf.reduce_sum(tf.square(s),axis=axis,
			keep_dims=True)
		safe_norm = tf.sqrt(squared_norm + epsilon)
		squash_factor = squared_norm / (1. + squared_norm)
		unit_vector = s / safe_norm
		return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")

caps2_n_caps = 10
caps2_n_dims = 16

init_sigma = 0.1

W_init = tf.random_normal(
		shape=(1,caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init,name="W")


W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
										name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
									 name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1,1,caps2_n_caps,1,1],
							name="caps1_output_tiled")
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
							name="caps2_predicted")
# Bij's
raw_weights = tf.zeros([batch_size, caps1_n_caps,caps2_n_caps,1,1],
						dtype=np.float32, name="raw_weights")
# Computes the Cij's
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights,caps2_predicted,
									name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions,axis=1, keep_dims=True,
							name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum,axis=-2,
							name="caps2_output_round_1")

caps2_output_round_1_tiled = tf.tile(
	caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
	name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
						transpose_a =True, name="agreement")
raw_weights_round_2 = tf.add(raw_weights, agreement,
							name="raw_weights_round_2")
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
										dim=2,
										name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
											caps2_predicted,
											name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
									axis=1, keep_dims=True,
									name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
							axis=-2,
							name= "caps2_output_round_2")
# the following variable should have shape(None, 1, 10, 16, 1)
caps2_output = caps2_output_round_2

# Computes Length of vectors in Capsule
def safe_norm(s, axis=-1,epsilon=1e-7,keep_dims=False,name=None):
	with tf.name_scope(name,default_name="safe_norm"):
		squared_norm = tf.reduce_sum(tf.square(s),axis=axis,
									keep_dims=keep_dims)
		return tf.sqrt(squared_norm + epsilon)

y_prob = safe_norm(caps2_output,axis=-2,name="y_prob")
y_prob_squeezed = tf.squeeze(y_prob,axis=-1)

###-----We wnt to store the lengths of vectors and their indices
###-----The length becomes a part of the loss function 
###-----The index becomes the prediction
# k represents how many digits we are looking for
k = 2
# gathers the top k longest vectors from the capsule layer and their indices
top_k_capsules = tf.nn.top_k(y_prob_squeezed,k)
y_pred = top_k_capsules.indices


y_pred = tf.squeeze(y_pred)

S = tf.one_hot(y_pred, depth=10)
S = tf.reduce_sum(S, axis=1)


y_pred = tf.cast(y_pred,tf.int64)
# top_k_capsules.indices returns what numbers are predicted in the image
# top_k_capsules.values returns the lengths of the vectors that represent the predicted numbers


y = tf.placeholder(shape=[None,k], dtype=tf.int64, name='y')

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")
T = tf.reduce_sum(T,axis=1)

caps2_output_norm = safe_norm(caps2_output, axis=-2,keep_dims=True,
								name="caps2_output_norm")
present_error_raw = tf.square(tf.maximum(0.,m_plus - caps2_output_norm),
							name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1,10),
							name="present_error")

absent_error_raw = tf.square(tf.maximum(0.,caps2_output_norm - m_minus),
							name="absent_error")

absent_error = tf.reshape(absent_error_raw, shape=(-1,10),
						name="absent_error")


L = tf.add(T*present_error,lambda_ * (1.0 - T) * absent_error,
		name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L,axis=1),name="margin_loss")

mask_with_labels = tf.placeholder_with_default(False,shape=(),
											name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels,
								lambda: y,
								lambda: y_pred,
								name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
								depth=caps2_n_caps,
								name="reconstruction_mask")

reconstruction_mask = tf.reduce_sum(reconstruction_mask,axis=1)

reconstruction_mask_reshaped = tf.reshape(
				reconstruction_mask, [-1,1,caps2_n_caps,1,1],
				name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
						caps2_output,reconstruction_mask_reshaped,
						name="caps2_output_masked")

decoder_input = tf.reshape(caps2_output_masked,
			[-1,caps2_n_caps * caps2_n_dims],
			name="decoder_input")
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 56

with tf.name_scope("decoder"):
	hidden1 = tf.layers.dense(decoder_input, n_hidden1,
							activation=tf.nn.relu,
							name="hidden1")
	hidden2 = tf.layers.dense(decoder_input, n_hidden2,
							activation=tf.nn.relu,
							name="hidden2")
	decoder_output = tf.layers.dense(hidden2,n_output,
							activation=tf.nn.sigmoid,
							name="decoder_output")

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
					name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
									name="reconstruction_loss")
alpha = 0.0005

loss = tf.add(margin_loss,alpha*reconstruction_loss,name="loss")

correct = tf.equal(T,S, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32), name="accuracy")

optimizer =tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 1
restore_checkpoint = True

n_iterations_per_epoch = train_length // batch_size
n_iterations_validation = valid_length // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
	if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
		saver.restore(sess, checkpoint_path)
	else:
		init.run()
	for epoch in range(n_epochs):
		for iteration in range(1, n_iterations_per_epoch + 1):
			X_batch, y_batch = train.next_batch()
			# Run the training operation and measure the loss:
			_, loss_train = sess.run(
				[training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 56, 1]),
                y: y_batch,
                mask_with_labels: True})
		print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  	end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
		loss_vals = []
		acc_vals = []
		for iteration in range(1, int(n_iterations_validation) + 1):
			X_batch, y_batch = validation.next_batch()
			loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 56, 1]),
                               y: y_batch})
			loss_vals.append(loss_val)
			acc_vals.append(acc_val)
			print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
		loss_val = np.mean(loss_vals)
		acc_val = np.mean(acc_vals)
		print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
		if loss_val < best_loss_val:
			save_path = saver.save(sess, checkpoint_path)
			best_loss_val = loss_val
n_iterations_test = test_length // batch_size
with tf.Session() as sess:
	saver.restore(sess, checkpoint_path)

	loss_tests = []
	acc_tests = []
	for iteration in range(1, n_iterations_test + 1):
		X_batch, y_batch = test.next_batch()
		loss_test, acc_test = sess.run(
				[loss,accuracy],
				feed_dict={X: X_batch.reshape([-1,28,56,1]),
				y: y_batch})
		loss_tests.append(loss_test)
		acc_tests.append(acc_test)
		print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
			iteration, n_iterations_test,
			iteration*100 / n_iterations_test),
		end=" "*10)
		loss_test = np.mean(loss_tests)
		acc_test = np.mean(acc_tests)
		print("\rFinal test accuracy: {:.4f}% Loss: {:.6f}".format(
			acc_test * 100, loss_test))























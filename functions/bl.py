import tensorflow as tf
from tensorflow.contrib import layers

def baseline(x, params, is_training):	#this is where the net is made(comes to existance)
	x = layers.batch_norm(x, is_training=is_training)
	for i in range(4):
		x = layers.conv2d(x, 16 * (2 ** i), 3, 1, activation_fn=tf.nn.elu, normalizer_fn=layers.batch_norm if params.use_batch_norm else None, normalizer_params={'is_training': is_training})
		x = layers.max_pool2d(x, 2, 2)

	# just take two kind of pooling and then mix them, why not :)
	mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
	apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

	x = 0.5 * (mpool + apool)	#this will be a output from 2 kinds of pooling which would be used later as an input for the other layers
	# we can use conv2d 1x1 instead of dense
	x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
	x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)

	# again conv2d 1x1 instead of dense layer
	logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)	#this is the final layer of the net
	return tf.squeeze(logits, [1, 2])	#returning the final net

"""
 CNN - TensorFlow Keras Sequental API.
 CIFAR-10 picture dataset (32 X 32 pixels with [R, G, B] color values). 49k training, 1k validation, 10k testing.
    Filter Size: 3x3
    Number of Filters: 32, 64, 128
    Pooling: 2x2 max-pooling
    Normalization: Batch normalization
    Network Architecture: (conv - batchnorm - relu - dropout - max pool) * 3 - avg pool - fc* 3
    Global average pooling: used instead of flattening after the final convolutional layer
    Regularization: Dropout
    Optimizer: Adam
"""
import tensorflow as tf
import numpy as np
#import os
#import math
#import timeit
#import matplotlib.pyplot as plt

#%matplotlib inline
# ---------------------------------------------------------------------------------
# load up the CIFAR-10 pictures into memory
def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):

    # Load the raw CIFAR-10 dataset
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test

# If there are errors with SSL downloading involving self-signed certificates,
# it may be that your Python version was recently installed on the current machine.
# See: https://github.com/tensorflow/tensorflow/issues/10779
# To fix, run the command: /Applications/Python\ 3.7/Install\ Certificates.command
#   ...replacing paths as necessary.

NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# ----------------------------- End Data Collection ------------------------------------------
# Use CPU or GPU. How often to print results.
USE_GPU = False
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'
print_every = 100 # mutated further below
print('Using device: ', device)
# ---------------------------------------------------------------------------------
# Put the pics into some data sets (training, validation, testing)
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)
# ---------------------------------------------------------------------------------
# Training Loop
def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1):
    tf.reset_default_graph()
    with tf.device(device):
        # Construct the computational graph
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool, name='is_training')
        scores = model_init_fn(x, is_training)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1
# ---------------------------------------------------------------------------------
# check accuracy and print results
def check_accuracy(sess, dset, x, scores, is_training=None):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
# ---------------------------------------------------------------------------------
# CNN Architecture!
def model_init_fn(inputs, is_training):
    model = None
    num_classes = 10
    initializer = tf.variance_scaling_initializer(scale=2.0)

    conv1 = tf.layers.conv2d(inputs, 32, [3,3], [1,1], padding='same', kernel_initializer=initializer)
    bn1 = tf.layers.batch_normalization(conv1, training=is_training)
    relu1 = tf.nn.elu(bn1)
    drop1 = tf.layers.dropout(relu1)
    pool1 = tf.layers.max_pooling2d(drop1, [2,2], [2,2])

    conv2 = tf.layers.conv2d(pool1, 64, [3,3], [1,1], padding='valid', kernel_initializer=initializer)
    bn2 = tf.layers.batch_normalization(conv2, training=is_training)
    relu2 = tf.nn.elu(bn2)
    drop2 = tf.layers.dropout(relu2)
    pool2 = tf.layers.max_pooling2d(drop2, [2,2], [2,2])

    conv3 = tf.layers.conv2d(pool2, 128, [3,3], [1,1], padding='valid', kernel_initializer=initializer)
    bn3 = tf.layers.batch_normalization(conv3, training=is_training)
    relu3 = tf.nn.elu(bn3)
    drop3 = tf.layers.dropout(relu3)

    avg_pool = tf.layers.average_pooling2d(drop3, [5,5], [1,1])
    avg_pool = tf.layers.flatten(avg_pool)
    fc1 = tf.layers.dense(avg_pool, 50)
    fc2 = tf.layers.dense(fc1, 50)
    scores = tf.layers.dense(fc2, num_classes)
    return scores
# ---------------------------------------------------------------------------------
# Optimization
learning_rate = 1e-3
def optimizer_init_fn():
    optimizer = None
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer
# ---------------------------------------------------------------------------------
def test_model(model_init_fn):
    tf.reset_default_graph()
    with tf.device(device):
        x = tf.zeros((50, 32, 32, 3))
        scores = model_init_fn(x, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)
# ---------------------------------------------------------------------------------
#device = '/device:GPU:0'
print_every = 700
num_epochs = 10
train_part34(model_init_fn, optimizer_init_fn, num_epochs)
#test_model(model_init_fn)
# ---------------------------------------------------------------------------------

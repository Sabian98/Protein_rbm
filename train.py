import os
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
from utilsnn import get_random_block_from_data
import matplotlib.pyplot as plt
# from tfrbm import GBRBM
import prot2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'F:\GMU\Spring 19\CS 701\protein\protein_rbm', 'Directory for storing data')
flags.DEFINE_integer('epochs',1000, 'The number of training epochs')
flags.DEFINE_integer('batchsize', 64, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')


train_x,test_x=prot2.data()  #data imported
#print(train_x[0:1])

# ensure output dir exists
if not os.path.isdir('out'):
  os.mkdir('out')

# RBMs
rbmobject1 = RBM(222, 128, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.01)
rbmobject2 = RBM(128, 64, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.01)
rbmobject3 = RBM(64, 32, ['rbmw3', 'rbvb3', 'rbmhb3'], 0.01)
rbmobject4 = RBM(32, 2,   ['rbmw4', 'rbvb4', 'rbmhb4'], 0.01,tf.nn.tanh)

if FLAGS.restore_rbm:
  rbmobject1.restore_weights('./out/rbmw1.chp')
  rbmobject2.restore_weights('./out/rbmw2.chp')
  rbmobject3.restore_weights('./out/rbmw3.chp')
  rbmobject4.restore_weights('./out/rbmw4.chp')

# Autoencoder
autoencoder = AutoEncoder(222, [128, 64, 32, 2], [['rbmw1', 'rbmhb1'],
                                                    ['rbmw2', 'rbmhb2'],
                                                    ['rbmw3', 'rbmhb3'],
                                                    ['rbmw4', 'rbmhb4']], tied_weights=True)

iterations = int(len(train_x) / FLAGS.batchsize)

#data_xs=get_random_block_from_data(train_x,64)
#print (data_xs)
# Train First RBM
print('first rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    # batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    data_xs=get_random_block_from_data(train_x,64)
    #print (data_xs[0:1])
    rbmobject1.partial_fit(data_xs)
  #print(rbmobject1.compute_cost(train_x))
  # show_image("out/1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./out/rbmw1.chp')
#
print('second rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    # batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batchsize)
    # Transform features with first rbm for second rbm
    data_xs = get_random_block_from_data(train_x, 64)
    batch_xs = rbmobject1.transform(data_xs)#being transformed into funny things
    print (batch_xs[0])
    rbmobject2.partial_fit(batch_xs)
  #print(rbmobject2.compute_cost(rbmobject1.transform(train_x)))
  # show_image("out/2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./out/rbmw2.chp')
#
print('third rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    # Transform features
    data_xs = get_random_block_from_data(train_x, 64)
    batch_xs = rbmobject1.transform(data_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    rbmobject3.partial_fit(batch_xs)
  #print(rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(train_x))))
  #show_image("out/3rbm.jpg", rbmobject3.n_w, (25, 20), (25, 10))
rbmobject3.save_weights('./out/rbmw3.chp')
#
# print("second training done")
# Train Third RBM
print('fourth rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    data_xs = get_random_block_from_data(train_x, 64)
    # Transform features
    batch_xs = rbmobject1.transform(data_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    batch_xs = rbmobject3.transform(batch_xs)
    rbmobject4.partial_fit(batch_xs)
  #print(rbmobject4.compute_cost(rbmobject3.transform(rbmobject2.transform(rbmobject1.transform(train_x)))))

rbmobject4.save_weights('./out/rbmw4.chp')


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./out/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
autoencoder.load_rbm_weights('./out/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)
autoencoder.load_rbm_weights('./out/rbmw3.chp', ['rbmw3', 'rbmhb3'], 2)
autoencoder.load_rbm_weights('./out/rbmw4.chp', ['rbmw4', 'rbmhb4'], 3)


print('autoencoder')
# cost=[]
for i in range(FLAGS.epochs):
  cost = 0.0
  for j in range(iterations):
    data_xs = get_random_block_from_data(train_x, 28)
    cost += autoencoder.partial_fit(data_xs)
  # print(cost)

print(autoencoder.transform(test_x))
#
#
#
#
#
#

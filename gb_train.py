import os
# from rbm import RBM

from au import AutoEncoder
import tensorflow as tf
from utilsnn import get_random_block_from_data
import matplotlib.pyplot as plt
# from tfrbm import GBRBM
import prot2
import matplotlib.pyplot as plt
from tfrbm import  GBRBM

train_x,test_x=prot2.data()

gbrbm1 = GBRBM(n_visible=222, n_hidden=300, learning_rate=0.3, momentum=0.95)
gbrbm2 = GBRBM(n_visible=300, n_hidden=100, learning_rate=0.3, momentum=0.95)
gbrbm3 = GBRBM(n_visible=100, n_hidden=8, learning_rate=0.3, momentum=0.95)
gbrbm4 = GBRBM(n_visible=8, n_hidden=2, learning_rate=0.3, momentum=0.95)
# gbrbm5 = GBRBM(n_visible=8, n_hidden=6, learning_rate=0.3, momentum=0.95)


iterations = int(len(train_x) / 64)
#GBRBM train
# errs1 = gbrbm1.fit(train_x, n_epoches=50, batch_size=32)
# tr1=gbrbm1.transform(train_x)
# gbrbm1.save_weights('./out/rbmw1.chp','w1')

print('first rbm')
for i in range(50):
  for j in range(iterations):

    data_xs=get_random_block_from_data(train_x,64)

    gbrbm1.partial_fit(data_xs)

gbrbm1.save_weights('./out/rbmw1.chp','w1')

#//
print('second rbm')
for i in range(50):
  for j in range(iterations):

    data_xs=get_random_block_from_data(train_x,64)
    tr2 = gbrbm1.transform(data_xs)
    gbrbm2.partial_fit(tr2)

gbrbm2.save_weights('./out/rbmw2.chp','w2')


print('third rbm')
for i in range(50):
  for j in range(iterations):

    data_xs=get_random_block_from_data(train_x,64)
    tr2 = gbrbm1.transform(data_xs)
    tr3=gbrbm2.transform(tr2)
    gbrbm3.partial_fit(tr3)

gbrbm3.save_weights('./out/rbmw3.chp','w3')


print('fourth  rbm')
for i in range(50):
  for j in range(iterations):

    data_xs=get_random_block_from_data(train_x,64)
    tr2=gbrbm1.transform(data_xs)
    tr3=gbrbm2.transform(tr2)
    tr4=gbrbm3.transform(tr3)
    gbrbm4.partial_fit(tr4)

gbrbm4.save_weights('./out/rbmw4.chp','w4')

# print('fifth rbm')
# for i in range(50):
#   for j in range(iterations):
#
#     data_xs=get_random_block_from_data(train_x,64)
#     tr2=gbrbm1.transform(data_xs)
#     tr3=gbrbm2.transform(tr2)
#     tr4=gbrbm3.transform(tr3)
#     tr5=gbrbm4.transform(tr4)
#     gbrbm5.partial_fit(tr5)
#
# gbrbm4.save_weights('./out/rbmw5.chp','w5')

#
tr1=gbrbm1.transform(train_x)

tr2=gbrbm2.transform(tr1)
tr3=gbrbm3.transform(tr2)
tr4=gbrbm4.transform(tr3)
# tr5=gbrbm5.transform(tr4)
print (tr4[0:23])
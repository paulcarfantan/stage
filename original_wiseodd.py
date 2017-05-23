#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
original_wiseodd.py

Created on Thu May 18 09:25:50 2017

@author: paul
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100 
print('\n Z_dim = ',Z_dim," (?)")
X_dim = mnist.train.images.shape[1]
print('\n X_dim = ',X_dim," (taille d'une image)")
y_dim = mnist.train.labels.shape[1]
print('\n y_dim = ',y_dim," (taille d'une liste de probabilités de labels) \n")
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)    #standard deviation of the normal distribution
    return tf.random_normal(shape=size, stddev=xavier_stddev)  #donne des valeurs aléatoires d'une distribution normale


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim])) 
# X_dim + y_dim plutôt que X_dim comme ça on train non seulement à dire si l'image appartient ou non au dataset original, MAIS AUSSI dire à quel label elle correspond !
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]  # Parameters to optimize
# D_W1 : shape=(794,128)  ;  D_W2 : shape=(128,1)  Weights pour passer d'un layer à l'autre  (input -->D_W1--> relu -->D_W2--> result)
# D_b1 : shape=(128,)  ;  D_b2 : shape=(1,)    Biases

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)    # Single layer Network
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit    # /!\ not scalar : operations /!\


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]
#G_W1 : shape=(110,128)  ;  G_W2 : shape=(128,784)
#G_b1 : shape=(128,)  ;  G_b2 : shape=(784,)

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)    # Z = random input images to begin with.
D_real, D_logit_real = discriminator(X, y)   # X = real dataset
D_fake, D_logit_fake = discriminator(G_sample, y)  # G_sample = generated dataset (fake)
# D_real & D_fake = unused   (D_fake = probability G fools D)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# labels = matrice de 1, même type que logits
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
# G_loss est basée sur D_logit_fake !

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
start = time.time()
zerotime = time.time()
num_it = 100000

for it in range(num_it):
    if it % 1000 == 0:

        delta = time.time() - start
        start = time.time()
        n_sample = 16

        Z_sample = sample_Z(n_sample, Z_dim)   # random matrix shape (16,100)
       # y_sample = np.ones(shape=[n_sample, y_dim]) * 0.1
        y_sample = np.zeros(shape=[n_sample, y_dim])
       # y_sample[:, 8] = 1
       # y_sample=np.random.rand(n_sample, y_dim)
       # y_sample=np.transpose(y_sample)/np.sum(y_sample,axis=1)
       # y_sample=np.transpose(y_sample)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        for k in range(n_sample): 
            y_sample[k,np.random.randint(0,y_dim)]=1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})    # samples.shape = (16 , 784)   => une colonne par image, n_mb lignes

        fig = plot(samples)
        plt.savefig('out/TEST4/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = mnist.train.next_batch(mb_size)
    # y_mb = labels des images du dataset original

    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    
    
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        #print('y = ',y)
        if it != 0 and it != num_it:
            t = (num_it - it) * delta / 1000
            print('time since last printing : ',delta,' sec)')
            print('ends approximately in ',t,' sec')
            print('( = ',t / 60,' min )')
            print((start - zerotime) * (num_it - it) / it,' sec')
        print('y_sample = ',y_sample)
        #print('samples[1,:].shape = ',samples[1,:].shape)
        #print('samples[1,:] = ',samples[1,:])
        print('samples.shape = ',samples.shape)
        print('samples[1,:] sum = ',sum(samples[1,:]))
        print('samples[:,1] sum = ',sum(samples[:,1]))
        #print('X_mb.shape = ',X_mb.shape)
        #print('y_mb.shape = ',y_mb.shape)
        #print('y_mb = ',y_mb)   # y_mb = labels de toutes les images (64) du batch => [0,0,0,0,0,0,0,1,0,0] = label 7
        print()

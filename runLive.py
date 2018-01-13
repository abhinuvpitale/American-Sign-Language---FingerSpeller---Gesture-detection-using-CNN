import tensorflow as tf
import parameters as par
import cv2
import numpy as np
saver = tf.train.import_meta_graph(par.saved_path+str('501.meta'))
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./Saved/'))

    # Get Operations to restore
    graph = sess.graph

    # Get Input Graph
    X = graph.get_tensor_by_name('Input:0')
    Y = graph.get_tensor_by_name('Target:0')
    keep_prob = tf.placeholder(tf.float32)

    # Get Ops
    prediction = graph.get_tensor_by_name('prediction:0')
    logits = graph.get_tensor_by_name('logits:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')






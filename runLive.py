import tensorflow as tf
import parameters as par

saver = tf.train.import_meta_graph(par.saved_path+str('500.meta'))
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./Saved/'))
    graph = tf.get_default_graph
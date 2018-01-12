import tensorflow as tf
import parameters as par
import util
import layers
import numpy as np


# Input Graph
X = tf.placeholder(tf.float32,[None,par.image_size,par.image_size,par.dim], name="Input")
Y = tf.placeholder(tf.uint8,[None,par.num_classes], name="Target")
keep_prob = tf.placeholder(tf.float32)

# Tensorboard stuff
global_step = tf.placeholder(tf.int64, shape=[], name="Step")

weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,par.dim,par.layer1])),
    'wc2': tf.Variable(tf.random_normal([5,5,par.layer1,par.layer2])),
    'wd1': tf.Variable(tf.random_normal([par.weirdlayershape,par.fc1])),
    'out': tf.Variable(tf.random_normal([par.fc1,par.num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([par.layer1])),
    'bc2': tf.Variable(tf.random_normal([par.layer2])),
    'bd1': tf.Variable(tf.random_normal([par.fc1])),
    'out': tf.Variable(tf.random_normal([par.num_classes]))
}

# Creating the model
logits = layers.convnet(X,weights,biases,keep_prob)
prediction = tf.nn.softmax(logits)

# Defining the loss
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y), name="Loss")
optimiser = tf.train.AdamOptimizer(par.learning_rate)
training_op =  optimiser.minimize(loss_op)

# Evaluating Model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

tf.summary.scalar(name="Loss", tensor=loss_op)
tf.summary.scalar(name="Accuracy", tensor=accuracy)
tf.summary.image(name="Input_images", tensor=X)
summary = tf.summary.merge_all()

saver = tf.train.Saver()

# Initializer
init = tf.global_variables_initializer()
saver_step = 100
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir="./Tensorboard", graph=sess.graph)

    sess.run(init)

    for step in range(1,par.num_steps+1):
        # Get the next batch
        batch_x,batch_y = util.get_next_batch(batch_size=par.batch_size, image_size=par.image_size, threshold=par.threshold)
        batch_y = tf.one_hot(np.reshape(batch_y, [-1]),par.num_classes)
        _, s = sess.run([training_op, summary],feed_dict={X:batch_x,Y:sess.run(batch_y),keep_prob:par.dropout, global_step: step})

        writer.add_summary(s, step)

        if step%par.display_step == 0 or step == 1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:sess.run(batch_y),keep_prob:1.0})
            print('Step '+str(step)+' Loss:'+str(loss)+' Accuracy: '+str(acc))
        if step%saver_step == 0:
            saver.save(sess,save_path=par.saved_path+str(step))

    print('Optimised!!')

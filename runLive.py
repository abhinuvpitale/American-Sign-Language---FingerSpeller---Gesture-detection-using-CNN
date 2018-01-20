import tensorflow as tf
import parameters as par
import cv2
import numpy as np
from PIL import ImageOps, Image

saver = tf.train.import_meta_graph(par.saved_path + str('501.meta'))
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Saved/'))

    # Get Operations to restore
    graph = sess.graph

    # Get Input Graph
    X = graph.get_tensor_by_name('Input:0')
    #Y = graph.get_tensor_by_name('Target:0')
    # keep_prob = tf.placeholder(tf.float32)
    keep_prob = graph.get_tensor_by_name('Placeholder:0')

    # Get Ops
    prediction = graph.get_tensor_by_name('prediction:0')
    logits = graph.get_tensor_by_name('logits:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')

    # Get the image

    while 1:
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        if ret:
            cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_img = img[100:300, 100:300]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            value = (35, 35)
            blurred = cv2.GaussianBlur(grey, value, 0)
            _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # cv2.imshow('title',thresh1)
            thresh1 = (thresh1 * 1.0) / 255
            thresh1 = Image.fromarray(thresh1)
            thresh1 = ImageOps.fit(thresh1, [par.image_size, par.image_size])
            if par.threshold:
                testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 1])
            else:
                testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 3])
            testImage = testImage.astype(np.float32)
            testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
            print(testY)
        else:
            continue

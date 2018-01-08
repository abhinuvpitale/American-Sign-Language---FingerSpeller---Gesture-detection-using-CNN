import tensorflow as tf
import parameters
import os
import numpy as np
from PIL import ImageOps,Image

# get batch of images
def get_next_batch(batch_size=128, image_size=30, threshold = parameters.threshold):
    allFiles = os.listdir(parameters.path)
    imgFiles = []
    for letter in allFiles:
        for image in os.listdir(parameters.path+'/'+letter):
            if image.endswith('.jpg'):
                absPath = parameters.path + '/' + letter + '/' + image
                if threshold:
                    if 'thresh' in image:
                        imgFiles.append(absPath)
                else:
                    if 'original' in image:
                        imgFiles.append(absPath)

    idx = np.random.permutation(len(imgFiles))
    idx = idx[0:batch_size]

    images = []
    labels = []

    for item in idx:
        # Get Image
        images.append(np.array(ImageOps.fit(Image.open(imgFiles[item]),[image_size,image_size]))/255)
        # Get Labels
        labels.append(parameters.letter[imgFiles[item].split('_')[-1][0]])

    if threshold:
        images = np.reshape(images,[-1, image_size, image_size, 1])
    else:
        images = np.reshape(images, [-1, image_size, image_size, 3])
    labels = np.array(labels)
    labels = np.reshape(labels,[-1,1])
    return [images.astype(np.float32),labels]


# Test the Function
# get_next_batch()
# get_next_batch(threshold=True)
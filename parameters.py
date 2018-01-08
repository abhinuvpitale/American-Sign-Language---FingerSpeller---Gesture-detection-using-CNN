path = './Datasets/osd/Dataset'
letter = {
    'a':0,
    'b':1,
    'c':2,
    'd':3,
    'g':4,
    'i':5,
    'l':6,
    'v':7,
    'y':8,
}
threshold = True


image_size = 32
num_classes = len(letter)
dropout = 0.5
display_step = 1
learning_rate = 0.001

# Number of parameters for each layer
layer1 = 32
layer2 = 64
fc1 = 1024

weirdlayershape = int(image_size*image_size/16)*layer2

learning_rate = 0.001
num_steps = 500
batch_size = 128

if threshold:
    dim = 1
else:
    dim = 3
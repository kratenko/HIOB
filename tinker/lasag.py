"""
Created on 2017-04-12

"""

pp = "/data/Peer/net/vgg16.pkl"

try:
    import cPickle as pickle
except ImportError:
    # Python 3
    import pickle
    with open(pp, 'rb') as f:
        model = pickle.load(f, encoding='latin-1')
else:
    # Python 2
    with open(pp, 'rb') as f:
        model = pickle.load(f)

weights = model['param values']  # list of network weight tensors
classes = model['synset words']  # list of class names
mean_pixel = model['mean value']  # mean pixel value (in BGR)
del model

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
net['pool1'] = PoolLayer(net['conv1_2'], 2)
net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2_2'], 2)
net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
net['pool3'] = PoolLayer(net['conv3_3'], 2)
net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
net['pool4'] = PoolLayer(net['conv4_3'], 2)
net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5_3'], 2)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
net['prob'] = NonlinearityLayer(net['fc8'], softmax)

lasagne.layers.set_all_param_values(net['prob'], weights)

import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'jpeg'
#%matplotlib inline

#from urllib.request import urlopen
from urllib import urlopen


import io
import skimage.transform


def prepare_image(url):
    ext = url.rsplit('.', 1)[1]
    img = plt.imread(io.BytesIO(urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = img.shape
    if h < w:
        img = skimage.transform.resize(
            img, (256, w * 256 / h), )  # preserve_range=True)
    else:
        img = skimage.transform.resize(
            img, (h * 256 / w, 256), )  # preserve_range=True)
    # Central crop to 224x224
    h, w, _ = img.shape
    img = img[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]
    # Remember this, it's a single RGB image suitable for plt.imshow()
    img_original = img.astype('uint8')
    # Shuffle axes from 01c to c01
    img = img.transpose(2, 0, 1)
    # Convert from RGB to BGR
    img = img[::-1]
    # Subtract mean pixel value
    img = img - mean_pixel[:, np.newaxis, np.newaxis]
    # Return the original and the prepared image (as a batch of a single item)
    return img_original, lasagne.utils.floatX(img[np.newaxis])


import theano
import theano.tensor as T


def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['fc8'], deterministic=True)
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])


def show_images(img_original, saliency, max_class, title):
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    max_class = max_class[0]
    # convert saliency from BGR to RGB, and from c01 to 01c
    saliency = saliency[::-1].transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    plt.show()


url = 'http://farm5.static.flickr.com/4064/4334173592_145856d89b.jpg'
img_original, img = prepare_image(url)

saliency_fn = compile_saliency_function(net)
saliency, max_class = saliency_fn(img)
show_images(img_original, saliency, max_class, "default gradient")

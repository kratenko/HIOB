from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne import layers
from nolearn.lasagne import NeuralNet

import theano.tensor as T


import lasagne
from PIL import Image
import numpy as np
import theano.tensor

import matplotlib.pyplot as plt
import math
import pickle

MAX_GUESS = 10
#image_path = '/data/Peer/ILSVRC2012_val_00040013.JPEG'
image_path = '/data/Peer/img5.JPEG'
image_path = '/data/Peer/Jogging/img/0027.jpg'
pretrained_path = "/data/Peer/vgg16.pkl"


def build_pmodel():
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('incept1', layers.Conv2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('output', layers.MaxPool2DLayer)
        ],
        input_shape=(None, 512, 28, 28),
        incept1_num_filters=20, incept1_filter_size=1, incept1_pad=0,
        conv2_num_filters=1, conv2_filter_size=3, conv2_pad=1,
        output_pool_size=(1, 1),
    )
    return net
    net = NeuralNet(
        layers=[
            (layers.InputLayer, {'shape': (None, 512, 28, 28)}),
            (layers.Conv2DLayer, {
             'num_filters': 20, 'filter_size': 1, 'pad': 0, }),
            (layers.Conv2DLayer, {
             'num_filters': 1, 'filter_size': 3, 'pad': 3, }),
        ],
    )
    return net
    net = {}
    net['input'] = InputLayer((None, 512, 28, 28), input_var=input_var)
    net['incept1'] = ConvLayer(net['input'], 20, 1, pad=0, )
    net['conv2'] = ConvLayer(net['incept1'], 1, 3, pad=1, )
    # net['conv3'] = ConvLayer(net['conv2'], 1, 3, pad=1, )
#     net['fc2'] = DenseLayer(net['incept1'], num_units=224 * 224)
    return net


def build_rmodel(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, )
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, )
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, )
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, )
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, )
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, )
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, )
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, )
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, )
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, )
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, )
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, )
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, )
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    return net


def load_image_pil(path, mean):
    """
    Load image file and prepare it for use in VGG16 net.
    """
    # load image
    im1 = Image.open(path)
    fac = im1.size[0] / 224, im1.size[1] / 224
    # resize
    im224 = im1.resize((224, 224))
    # convert to numpy:
    imnp = np.array(im224.getdata(), dtype="float32")
    # rgb to bgr conversion:
    for xs in imnp:
        xs[0], xs[2] = xs[2], xs[0]
    # reshaping to 2d image
    imnp.shape = (224, 224, 3)
    # apply mean
    for n in range(3):
        imnp[:, :, n] -= mean[n]
    # do whatever:
    imnp = imnp.transpose((2, 0, 1))
    imnp = np.expand_dims(imnp, axis=0)
    # done
    return im224, imnp, fac

"""
f = plt.figure()

im0 = Image.open(image_path)
im1 = im0.resize((224, 224))

f.add_subplot(1, 2, 1)
imgplot = plt.imshow(im0)
f.add_subplot(1, 2, 2)
imgplot = plt.imshow(im1)
plt.show()


exit()
"""

print("building net...")
input_var = theano.tensor.tensor4('input_var')
output_var = theano.tensor.fmatrix('output_var')
net = build_rmodel(input_var)

print("loading trained model")
f = np.load("conv5.npz")
# print f.keys()
params = [f["arr_%d" % i] for i in range(len(f.files))]
f.close()

print("Setting params to model")
lasagne.layers.set_all_param_values(net['pool5'], params)

mean = [103.939, 116.779, 123.68]
print("loading image '%s'..." % image_path)
im, ndat, fac = load_image_pil(image_path, mean)
print(ndat.shape)

gt = (174, 93, 205, 204)
print(gt)
print(fac)
gtc = (gt[0] / fac[0], gt[1] / fac[1], gt[2] / fac[0], gt[3] / fac[1])
print(gtc)

print("fetching layers")
conv4_3 = net['conv4_3']
conv5_3 = net['conv5_3']
pool4 = net['pool4']
pool5 = net['pool5']

pred_conv = lasagne.layers.get_output((conv4_3, conv5_3), deterministic=True)
# print pred_conv, pool5

print("Testing image on Net")
conv_prediction = theano.function([input_var], pred_conv)
pp = conv_prediction(ndat)
print("Plotting...")
print(pp[0].shape, pp[1].shape)

data4all = pp[0]
data4 = pp[0][0]
data5all = pp[1]
data5 = pp[1][0]
print(data4.shape, data5.shape)


# fore ground mask:
fm1 = np.zeros(shape=(224, 224), dtype="float32")
rect = (int(math.floor(gtc[0])), int(math.floor(gtc[1])), int(
    math.ceil(gtc[2])), int(math.ceil(gtc[3])))
print(rect)
#fm1[rect[0]:rect[2], rect[1]:rect[3]] = 1
fm1[rect[1]:rect[3], rect[0]:rect[2]] = 1
#fm1 = fm1.resize(28, 28)
fm1l = fm1.reshape(224 * 224)
fm28 = fm1.reshape((28, int(224 / 28), 28, int(224 / 28))).mean(3).mean(1)
fm28x = fm28.reshape((1, 1, 28, 28))
print("shapes:", fm1.shape, fm1l.shape, data4.shape, fm28.shape, fm28x.shape)

# hi
for i in range(100):
    img = plt.imshow(data5[0 + i], cmap='hot', interpolation='nearest')
#img2 = plt.imshow(data5[31], cmap='hot', interpolation='nearest')
    plt.show()
exit()


# p-net
pnet = build_pmodel()
print(pnet)
pnet.initialize()
# pnet.initialize_layers()
o = pnet.get_output('output', data4all)
print(o)
#pnet.fit(data4all, fm28x)
exit()

# train pnet
pnet_output = pnet['conv2']
pnet_prediction = lasagne.layers.get_output(pnet_output)
pnet_target = T.tensor4('targets')
pnet_loss = lasagne.objectives.categorical_crossentropy(
    pnet_prediction, pnet_target)
pnet_loss = pnet_loss.mean()
#
pnet_params = lasagne.layers.get_all_param_values(pnet_output, trainable=True)
pnet_updates = lasagne.updates.nesterov_momentum(
    pnet_loss, pnet_params, learning_rate=0.01, momentum=0.9)
exit()


num = 10
f = plt.figure()
f.add_subplot(2, 6, 1)
plt.imshow(im)
f.add_subplot(2, 6, 2)
img = plt.imshow(fm1, cmap='hot', interpolation='nearest')

for i in range(num):
    f.add_subplot(2, 6, i + 3)
    img = plt.imshow(data4[i], cmap='hot', interpolation='nearest')

plt.show()
exit()

for i in range(100):
    img = plt.imshow(data5[200 + i], cmap='hot', interpolation='nearest')
#img2 = plt.imshow(data5[31], cmap='hot', interpolation='nearest')
    plt.show()
# exit()

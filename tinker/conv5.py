from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

import lasagne
from PIL import Image
import numpy as np
import theano.tensor

import matplotlib.pyplot as plt

MAX_GUESS = 10
#image_path = '/data/Peer/ILSVRC2012_val_00040013.JPEG'
image_path = '/data/Peer/img5.JPEG'
image_path = '/data/Peer/Jogging/img/0027.jpg'
pretrained_path = "/data/Peer/vgg16.pkl"


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
    return imnp


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
ndat = load_image_pil(image_path, mean)
print(ndat.shape)

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

data4 = pp[0][0]
data5 = pp[1][0]
print(data4.shape, data5.shape)

for i in range(100):
    img = plt.imshow(data5[0 + i], cmap='hot', interpolation='nearest')
#img2 = plt.imshow(data5[31], cmap='hot', interpolation='nearest')
    plt.show()
# exit()

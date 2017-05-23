from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

import pickle
import lasagne
from PIL import Image
import numpy as np
import theano.tensor


MAX_GUESS = 10
#image_path = '/data/Peer/ILSVRC2012_val_00040013.JPEG'
image_path = '/data/Peer/img19.JPEG'
pretrained_path = "/data/Peer/vgg16.pkl"


def build_model(input_var):
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
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

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
        imnp[:,:,n] -= mean[n]
    # do whatever:
    imnp = imnp.transpose((2,0,1))
    imnp = np.expand_dims(imnp, axis=0)
    # done
    return imnp


print("loading model...")
with open(pretrained_path, 'rb') as mfile:
    model = pickle.load(mfile, encoding='latin1')
#model = pickle.load(open(pretrained_path))
print(model.keys())
mean_image = model['mean value']
classes = model['synset words']
# print mean_image

print("loading image '%s'..." % image_path)
ndat = load_image_pil(image_path, mean_image)

values = model['param values']
print("building net...")
input_var = theano.tensor.tensor4('input_var')
output_var = theano.tensor.fmatrix('output_var')
net = build_model(input_var)

print("setting net to model")
output_layer = net['prob']
lasagne.layers.set_all_param_values(output_layer, values)

print("Testing image on Net")
prediction = lasagne.layers.get_output(output_layer, deterministic=True)
output_prediction = theano.function([input_var], prediction)
pred = output_prediction(ndat)

guesses = pred[0].argsort()[-MAX_GUESS:][::-1]
print(guesses)
print("Best guesses:")
for n, guess in enumerate(guesses):
    print("%02d: (%03d/%6.4f) %s" % (n+1, guess, pred[0][guess], classes[guess]))
    #print guess, pred[0][guess], classes[guess]

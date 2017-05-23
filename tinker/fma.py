"""
Created on 2016-08-18

@author: Peer Springstübe
"""

import logging

import lasagne.layers
import theano.tensor

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import math


class Vgg16Input(object):
    _instance_counter = 0

    def __init__(self, image, input_size, mean):
        Vgg16Input._instance_counter += 1
        self.number = Vgg16Input._instance_counter

        self.original_ground_truth = None
        self.image_path = None
        self.original_size = image.size
        self.original_image = image
        self.input_size = input_size
        self.input_image = self.original_image.resize(self.input_size)
        self.scale_factor = tuple(
            self.input_size[i] / self.original_size[i] for i in range(2))
        # convert to numpy:
        np_img = np.array(self.input_image.getdata(), dtype="float32")
        # rgb to bgr conversion:
        for xs in np_img:
            xs[0], xs[2] = xs[2], xs[0]
        # reshaping to 2d image
        np_img.shape = (self.input_size[0], self.input_size[1], 3)
        # apply mean
        self.mean = tuple(mean)
        for n in range(3):
            np_img[:, :, n] -= mean[n]
        # do whatever:
        np_img = np_img.transpose((2, 0, 1))
        np_img = np.expand_dims(np_img, axis=0)
        # store prepared data in array:
        self.input_array = np_img

    def elaborate(self):
        return """Vgg16Input #{self.number}
----------
image_path:    {self.image_path}
original_size: {self.original_size}
input_size:    {self.input_size}
scale_factor:  {self.scale_factor}
mean:          {self.mean}
original_gt:   {self.original_ground_truth}
        """.format(self=self)


class Vgg16Prediction(object):

    def __init__(self, names, predictions):
        self.sample = None
        self.net = None
        self.layers = dict(zip(names, predictions))


class Vgg16F(object):
    logger = logging.getLogger(__name__)

    def __init__(self, input_size=None):
        if input_size is None:
            input_size = (224, 224)
        self.layers = []
        self.input_size = input_size
        self.input_var = theano.tensor.tensor4()
        self.build_net()
        self.mean = [103.939, 116.779, 123.68]
        self.prediction_layer_names = ['conv4_3', 'conv5_3']
        self.prediction_layers = None
        self.prediction_outputs = None
        self.prediction_function = None

    def build_net(self):
        self.logger.info(
            "Building Vgg16F-net with size %dx%d", *self.input_size)
        InputLayer = lasagne.layers.InputLayer
        ConvLayer = lasagne.layers.Conv2DLayer
        PoolLayer = lasagne.layers.Pool2DLayer
        net = {}
        net['input'] = InputLayer(
            (None, 3, self.input_size[0], self.input_size[1]), input_var=self.input_var)
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
        self.layers = net

    def load_model(self, path="conv5.npz"):
        self.logger.info("loading trained model from file '%s'", path)
        with np.load(path) as f:
            params = [f["arr_%d" % i] for i in range(len(f.files))]

        self.logger.info("Setting loaded parameters to net")
        lasagne.layers.set_all_param_values(self.layers['pool5'], params)
        self.logger.info("model loaded")

    def load_sample(self, path):
        self.logger.info("loading image from file '%s'", path)
        im = Image.open(path)
        sample = Vgg16Input(im, self.input_size, self.mean)
        sample.image_path = path
        return sample

    def _build_prediction_function(self):
        if self.prediction_function is not None:
            # we so this only once
            return

        self.logger.info("Building prediction function")
        # Pick layers that will be included in the prediction by name
        self.prediction_layers = [self.layers[name]
                                  for name in self.prediction_layer_names]
        # Tell lasagne what outputs we want:
        self.prediction_outputs = lasagne.layers.get_output(
            self.prediction_layers, deterministic=True)
        # Create Theano-function for getting those outputs
        self.prediction_function = theano.function(
            [self.input_var], self.prediction_outputs)

    def predict_sample(self, sample):
        # make sure prediction function has been created:
        self._build_prediction_function()
        # calculate prediction for all requested layers:
        self.logger.info("Predicting sample")
        prediction_data = self.prediction_function(sample.input_array)
        # wrap prediction in object:
        prediction = Vgg16Prediction(
            self.prediction_layer_names, prediction_data)
        # set oder stuff:
        prediction.sample = sample
        prediction.net = self
        # return Prediction-instance:
        return prediction


class FeatureAnalyser(object):

    def _build_fg_map(self, size, gt):
        m = np.zeros(shape=size, dtype='float32')
        outer_rect = math.floor(gt[0]), math.floor(gt[1]), math.ceil(
            gt[0] + gt[2]), math.ceil(gt[1] + gt[3])
        rect = tuple(int(c) for c in outer_rect)
        m[rect[1]:rect[3], rect[0]:rect[2]] = 1
        return m

    def __init__(self, prediction):
        self.prediction = prediction
        self.sample = prediction.sample
        self.fg_original = self._build_fg_map(
            self.sample.original_size, self.sample.original_ground_truth)


logging.basicConfig(level=logging.DEBUG)
n = Vgg16F(input_size=(368, 368))
n.load_model()
s = n.load_sample('/data/Peer/Jogging/img/0027.jpg')
s.original_ground_truth = (174, 93, 31, 111)
p = n.predict_sample(s)
fa = FeatureAnalyser(p)
print(fa.fg_original.shape, fa.fg_original)
exit()
# print(p.layers)
print(s.elaborate())
f = plt.figure()
f.add_subplot(5, 20, 1)
plt.imshow(s.input_image)
data = p.layers['conv5_3'][0]
print(data.shape)
num = 99
for i in range(num):
    f.add_subplot(5, 20, i + 2)
    img = plt.imshow(
        data[i + (num * 0)], cmap='hot', interpolation='nearest')
#    f.add_subplot(11, 6, 2)
print("plotting")
plt.show()

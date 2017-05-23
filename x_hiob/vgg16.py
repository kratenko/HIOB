# Built after from Lasagne's modelzoo under
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/
# Pretrained weights are located at:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import os
from collections import OrderedDict
import pickle
import logging

import numpy as np
import lasagne.layers
import theano.tensor

import x_hiob
from hiob.sample import Representation, Classification, FeatureMap

logger = logging.getLogger(__name__)


class Vgg16Preparation(Representation):
    """
    Sample preparation for Vgg16 net or Vgg16F net.

    Consists of resized image (typically 224x224x3) and input array. 
    """
    name = "Vgg16/input"
    image_description = "Resized RGB image"
    data_description = "Mean applied BGR image"

    def __init__(self, net, sample):
        super().__init__()
        # set references:
        self.sample = sample
        self.source = sample.representations['original']
        self.processor = net
        logger.info("Preparing sample data for Vgg16")
        self._generate_image()
        self._generate_input_data()
        self._generate_name()

    def _generate_name(self):
        self.name = self.processor.name + "/input"

    def _generate_image(self):
        self.size = self.processor.input_size
        logger.info("Resizing sample image from %dx%d to %dx%d...",
                    *(self.source.size + self.size))
        self.image = self.source.image.resize(self.size)

    def _generate_input_data(self):
        # convert to numpy
        logger.debug("Generating numpy-array from resized image...")
        np_image = np.array(self.image.getdata(), dtype="float32")
        # rgb to bgr conversion:
        logger.debug("Converting array from RGB to BGR...")
        for xs in np_image:
            xs[0], xs[2] = xs[2], xs[0]
        # reshaping to 2d image
        logger.debug("Reshaping array to 2d...")
        np_image.shape = (self.image.size[0], self.image.size[1], 3)
        # apply mean
        logger.debug("Applying mean to pixel data...")
        self.mean = self.processor.mean
        for n in range(3):
            np_image[:, :, n] -= self.mean[n]
        # do whatever:
        logger.debug("Finalizing shape of input array...")
        np_image = np_image.transpose((2, 0, 1))
        np_image = np.expand_dims(np_image, axis=0)
        # done
        self.data = np_image


class Vgg16(object):
    PRETRAINED_MODEL_URL = "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl"
    MODEL_FILE = "vgg16.pkl"
    name = "Vgg16"

    def __init__(self, name=None, use_dnn=False):
        """
        Create new Vgg16 network instance.
        """
        if name is None:
            self.name = "Vgg16"
        else:
            self.name = name
        logger.info("Creating new Vgg16-net: '%s'", self.name)
        self.use_dnn = use_dnn
        self.mean = None
        self.classes = None
        self.input_size = (224, 224)
        self.input_depth = 3
        self.input_shape = (None, self.input_depth) + self.input_size
        self.input_var = theano.tensor.tensor4()
        self.layers = {}
        self.build_net()

        self.prediction_layer_names = ['prob']
        self.prediction_layers = None
        self.prediction_outputs = None
        self.prediction_function = None

    def build_net(self):
        """
        Build layers of Vgg16 network.
        """
        logger.info("Creating layers for Vgg16")
        InputLayer = lasagne.layers.InputLayer
        PoolLayer = lasagne.layers.Pool2DLayer
        DenseLayer = lasagne.layers.DenseLayer
        DropoutLayer = lasagne.layers.DropoutLayer
        NonlinearityLayer = lasagne.layers.NonlinearityLayer
        if self.use_dnn:
            logger.info("Using DNN-Layers")
            ConvLayer = lasagne.layers.dnn.Conv2DDNNLayer
        else:
            logger.info("Not using DNN-Layers")
            ConvLayer = lasagne.layers.Conv2DLayer
        net = OrderedDict()
        net['input'] = InputLayer(
            self.input_shape, input_var=self.input_var)
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
        net['prob'] = NonlinearityLayer(
            net['fc8'], lasagne.nonlinearities.softmax)
        self.layers = net

    def _build_prediction_function(self):
        """
        Create theano function for prediction (if not done so, yet).
        """
        if self.prediction_function is not None:
            # we do this only once
            return

        logger.info("Building prediction function for '%s'", self.name)
        logger.debug("Predicted layers: %s", self.prediction_layer_names)
        # Pick layers that will be included in the prediction by name
        self.prediction_layers = [self.layers[name]
                                  for name in self.prediction_layer_names]
        # Tell lasagne what outputs we want:
        self.prediction_outputs = lasagne.layers.get_output(
            self.prediction_layers, deterministic=True)
        # Create Theano-function for getting those outputs
        self.prediction_function = theano.function(
            [self.input_var], self.prediction_outputs)

    def predict(self, data):
        """
        Calculate net's prediction for input data.
        """
        # make sure prediction function has been created:
        self._build_prediction_function()
        # calculate prediction for all requested layers:
        logger.info("Predicting on '%s'", self.name)
        prediction_data = self.prediction_function(data)
        return prediction_data

    def predict_sample(self, sample):
        """
        Calculate net's prediction and add to sample's representations.
        """
        # get preparation, create if needed:
        preparation = self.prepare_sample(sample)
        # execute net on data:
        prediction_data = self.predict(preparation.data)
        # extract classification layer:
        prop = prediction_data[0][0]
        cl = Classification(self, sample, prop)
        cl.source = sample.representations[self.name + '/input']
        cl.processor = self
        sample.representations[self.name + '/classification'] = cl
        return cl

    def download_model(self):
        x_hiob.touch_model_dir()
        logger.info("Downloading pretrained model for '%s' from '%s' to  '%s'",
                    self.name, self.PRETRAINED_MODEL_URL, self.model_path)
        from urllib.request import urlretrieve
        urlretrieve(self.PRETRAINED_MODEL_URL, self.model_path)

    def load_model(self):
        self.model_path = x_hiob.model_file_path(self.MODEL_FILE)
        if not os.path.exists(self.model_path):
            self.download_model()
        logger.info(
            "Loading model for '%s' from file '%s'", self.name, self.model_path)
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        mean_image = model['mean value']
        classes = model['synset words']
        param_values = model['param values']
        self.mean = mean_image
        self.classes = classes
        # load data in to net:
        lasagne.layers.set_all_param_values(self.layers['prob'], param_values)
        logger.debug(
            "'%s': mean=%s, classes: %d", self.name, self.mean, len(self.classes))

    def prepare_sample(self, sample):
        input_name = self.name + '/input'
        if input_name in sample.representations:
            logger.debug(
                "Required representation '%s' already in %s", input_name, sample.__str__())
            return sample.representations[input_name]
        logger.info(
            "Generating representation '%s' for %s", input_name, sample.__str__())
        preparation = Vgg16Preparation(self, sample)
        # store new preparation in sample entity:
        sample.representations[input_name] = preparation
        return preparation


class Vgg16F(object):
    """
    Convolutional layers of Vgg16 net only.
    """
    MODEL_FILE = "vgg16f.pkl"
    name = "Vgg16F"

    def __init__(self, name=None, input_size=None, use_dnn=False):
        """
        Create new Vgg16F network instance.
        """
        if name is None:
            self.name = "Vgg16F"
        else:
            self.name = name
        logger.info("Creating new Vgg16F-net: '%s'", self.name)
        self.use_dnn = use_dnn
        self.mean = None
        self.classes = None
        if input_size is None:
            self.input_size = (224, 224)
        else:
            self.input_size = input_size
        self.input_depth = 3
        self.input_shape = (None, self.input_depth) + self.input_size
        self.input_var = theano.tensor.tensor4()
        self.layers = {}
        self.build_net()

        self.prediction_layer_names = ['conv4_3', 'conv5_3']
        self.prediction_layers = None
        self.prediction_outputs = None
        self.prediction_function = None

    def build_net(self):
        """
        Build convolutional layers of Vgg16 network.
        """
        logger.info("Creating layers for Vgg16F")
        InputLayer = lasagne.layers.InputLayer
        PoolLayer = lasagne.layers.Pool2DLayer
        if self.use_dnn:
            logger.info("Using DNN-Layers")
            ConvLayer = lasagne.layers.dnn.Conv2DDNNLayer
        else:
            logger.info("Not using DNN-Layers")
            ConvLayer = lasagne.layers.Conv2DLayer
        net = OrderedDict()
        net['input'] = InputLayer(
            self.input_shape, input_var=self.input_var)
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

    def build_model(self):
        x_hiob.touch_model_dir()
        logger.info(
            "Building pretrained model for '%s' from Vgg16 model", self.name)
        # if vgg16-model is also missing, it is downloaded:
        vgg16 = Vgg16()
        vgg16.load_model()
        # extract last layer we need for this model:
        logger.info("Extracting needed layers from Vgg16 model")
        last_layer = vgg16.layers['pool5']
        values = lasagne.layers.get_all_param_values(last_layer)
        # storing all in convenient dictionary:
        model = {
            'mean value': vgg16.mean,
            'param values': values,
        }
        # pickle model dictionary to model file:
        logger.info("Saving Vgg16F model to file '%s'", self.model_path)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self):
        self.model_path = x_hiob.model_file_path(self.MODEL_FILE)
        if not os.path.exists(self.model_path):
            self.build_model()
        logger.info(
            "Loading model for '%s' from file '%s'", self.name, self.model_path)
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        mean_image = model['mean value']
        param_values = model['param values']
        self.mean = mean_image
        # load data in to net:
        lasagne.layers.set_all_param_values(self.layers['pool5'], param_values)
        logger.debug(
            "'%s': mean=%s", self.name, self.mean)

    def _build_prediction_function(self):
        """
        Create theano function for prediction (if not done so, yet).
        """
        if self.prediction_function is not None:
            # we do this only once
            return

        logger.info("Building prediction function for '%s'", self.name)
        logger.debug("Predicted layers: %s", self.prediction_layer_names)
        # Pick layers that will be included in the prediction by name
        self.prediction_layers = [self.layers[name]
                                  for name in self.prediction_layer_names]
        # Tell lasagne what outputs we want:
        self.prediction_outputs = lasagne.layers.get_output(
            self.prediction_layers, deterministic=True)
        # Create Theano-function for getting those outputs
        self.prediction_function = theano.function(
            [self.input_var], self.prediction_outputs)

    def prepare_sample(self, sample):
        input_name = self.name + '/input'
        if input_name in sample.representations:
            logger.debug(
                "Required representation '%s' already in %s", input_name, sample.__str__())
            return sample.representations[input_name]
        logger.info(
            "Generating representation '%s' for %s", input_name, sample.__str__())
        preparation = Vgg16Preparation(self, sample)
        # store new preparation in sample entity:
        sample.representations[input_name] = preparation
        return preparation

    def predict(self, data):
        """
        Calculate net's prediction for input data.
        """
        # make sure prediction function has been created:
        self._build_prediction_function()
        # calculate prediction for all requested layers:
        logger.info("Predicting on '%s'", self.name)
        prediction_data = self.prediction_function(data)
        return prediction_data

    def predict_sample(self, sample):
        """
        Calculate net's prediction and add to sample's representations.
        """
        # get preparation, create if needed:
        preparation = self.prepare_sample(sample)
        # execute net on data:
        prediction_data = self.predict(preparation.data)
        # extract classification layer:
        for n, layer_name in enumerate(self.prediction_layer_names):
            feature_data = prediction_data[n][0]
            feature_name = self.name + "/" + layer_name
            fmap = FeatureMap(self, sample, feature_name, feature_data)
            sample.representations[feature_name] = fmap
            # TODO: do something!

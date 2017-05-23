import PIL.Image
import logging
from collections import OrderedDict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Representation(object):
    """
    Superclass for all representations of sample.
    """
    source = None
    processor = None
    name = "[unknown]"

    def __init__(self):
        pass

    def __str__(self):
        return "<x_hiob.Representation:{}#{}>".format(self.name, self.sample.instance_number)


class Original(Representation):
    INTERNAL_IMAGE_MODE = "RGB"
    INTERNAL_IMAGE_DEPTH = 3

    def __init__(self):
        super().__init__()
        self.name = "original"

    def load_image_from_file(self, path):
        self.path = path
        im = PIL.Image.open(self.path)
        self.original_image_mode = im.mode
        if im.mode != self.INTERNAL_IMAGE_MODE:
            logger.info(
                "Converting image mode from '%s' to '%s'", im.mode, self.INTERNAL_IMAGE_MODE)
            self.image = im.convert(self.INTERNAL_IMAGE_MODE)
        else:
            logger.info("Image already in mode '%s'", self.INTERNAL_IMAGE_MODE)
            self.image = im
        # extract image meta data:
        self.size = tuple(self.image.size)
        self.depth = self.INTERNAL_IMAGE_DEPTH


class FeatureMap(Representation):

    def __init__(self, processor, sample, name, data):
        super().__init__()
        self.processor = processor
        self.sample = sample
        self.name = name
        self.data = data
        self.depth = data.shape[0]
        self.size = data.shape[1], data.shape[2]

    def plot(self, n):
        plt.imshow(self.data[n], cmap='hot', interpolation='nearest')


class Classification(Representation):
    classes = None

    def __init__(self, processor, sample, data):
        super().__init__()
        self.processor = processor
        self.sample = sample
        self.data = data
        self.classes = processor.classes
        self.source = sample.representations[processor.name + '/input']
        self.name = processor.name + '/classification'

    def best_guesses(self, number):
        if number > len(self.classes):
            raise ValueError(
                "Cannot get {} best guesses for {} classes.".format(number, len(self.classes)))
        # get indexes for `number` best guesses:
        guesses = self.data.argsort()[-number:][::-1]
        # create list of best guesses
        # [confidence, class-number, class-label]
        return [(self.data[guess], guess, self.classes[guess]) for guess in guesses]


class Sample(object):
    _instance_counter = 0

    def _set_instance_number(self):
        Sample._instance_counter += 1
        self.instance_number = Sample._instance_counter

    def __str__(self):
        return "<x_hiob.Sample#{}>".format(self.instance_number)

    def __init__(self):
        self._set_instance_number()
        logger.info("Creating new sample: %s", self)
        #
        self.representations = OrderedDict()

    def load_original_image(self, path):
        original = Original()
        original.load_image_from_file(path)
        self.representations['original'] = original

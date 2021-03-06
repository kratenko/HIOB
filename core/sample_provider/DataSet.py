"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import os
import logging
import re
from .Sample import Sample
from .FakeLiveSample import FakeLiveSample
from .DataSetException import DataSetException

logger = logging.getLogger(__name__)

path_sep_pattern = re.compile('/')


class InvalidSetNameError(BaseException):
    pass


class DataSet(object):

    def __init__(self, name, data_dir):
        self.name = name
        self.description = None
        self.samples = []
        self.samples_by_name = {}
        self.total_samples = 0
        self.format = None
        self.path = os.path.join(data_dir, name)

    def load(self, definition, tracking_conf=None):
        if self.name == '__ros__':
            raise InvalidSetNameError("'__ros__' is a reserved keyword. It is not allowed for sample names!")
        if 'description' in definition:
            self.description = definition['description']
        if 'format' in definition:
            self.format = definition['format']
        fake_fps = tracking_conf['fake_fps'] if 'fake_fps' in tracking_conf else 0
        skip_frames = tracking_conf['skip_frames'] if 'skip_frames' in tracking_conf else 0
        for sdef in definition['samples']:
            #print("fake_fps is {} and skip_frames is {}.".format(fake_fps, skip_frames))
            if fake_fps > 0 or skip_frames > 0:
                logger.info("Adding fake live-sample (fake_fps:{}, skip_frames:{})".format(fake_fps, skip_frames))
                s = FakeLiveSample(self, sdef['name'], fake_fps, skip_frames)
            else:
                logger.info("Adding sample")
                s = Sample(self, sdef['name'])
            if 'attributes' in sdef:
                s.attributes = tuple(sdef['attributes'])
            if 'first_frame' in sdef:
                s.first_frame = int(sdef['first_frame'])
            if 'last_frame' in sdef:
                s.last_frame = int(sdef['last_frame'])
            if 'actual_frames' in sdef:
                s.actual_frames = int(sdef['actual_frames'])
            else:
                raise DataSetException(
                    'No total_frames in sample {}/{}'.format(self.name, sdef['name']))
            self.samples.append(s)
            self.samples_by_name[s.name] = s
        self.total_samples = len(self.samples)

    def __repr__(self):
        return "<DataSet {name}, {total_samples} samples>".format(name=self.name, total_samples=self.total_samples)


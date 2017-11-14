"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import logging

logger = logging.getLogger(__name__)


class DataCollection(object):

    def __init__(self, directory, name):
        self.directory = directory
        self.name = name
        self.description = None
        self.samples_full_names = []
        self.samples_parsed = []
        self.total_samples = 0
        self.samples = []
        self.loaded = False

    def load(self, definition):
        if 'description' in definition:
            self.description = definition['description']
        for snam in definition['samples']:
            self.samples_full_names.append(snam)
            self.samples_parsed.append(snam.split('/'))
            self.total_samples += 1

    def load_samples(self, fake_fps=0):
        if self.loaded:
            return
        for p1, p2 in self.samples_parsed:
            if p1 == 'SET':
                # this is a full data set
                ds = self.directory.get_data_set(p2, fake_fps)
                self.samples.extend(ds.samples)
            elif p2 == 'COLLECTION':
                # this is a collection within a collection - not supported
                raise NotImplementedError(
                    "Collections within collections of samples are not supported.")
            else:
                set_name, sample_name = p1, p2
                self.samples.append(
                    self.directory.get_sample(set_name, sample_name, fake_fps))
        self.loaded = True


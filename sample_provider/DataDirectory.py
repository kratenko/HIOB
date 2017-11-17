"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import os
import logging

import yaml

from .DataSet import DataSet
from .DataCollection import DataCollection
from .LiveSample import LiveSample

logger = logging.getLogger(__name__)


class DataDirectory(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        conf_dir = 'conf'
        self.data_set_dir = os.path.join(conf_dir, 'data_sets')
        self.data_collection_dir = os.path.join(conf_dir, 'collections')

        self.data_sets = {}
        self.data_collections = {}

    def _load_data_set(self, name, fake_fps=0):
        set_file = os.path.join(self.data_set_dir, name + '.yaml')
        ds = DataSet(name, self.data_dir)
        with open(set_file, "r") as f:
            y = yaml.safe_load(f)
            ds.load(y, fake_fps)
        self.data_sets[name] = ds

    def _load_data_collection(self, name, fake_fps=0):
        col_file = os.path.join(self.data_collection_dir, name + '.yaml')
        dc = DataCollection(self, name)
        with open(col_file, "r") as f:
            y = yaml.safe_load(f)
            dc.load(y, fake_fps)
        self.data_collections[name] = dc

    def get_data_set(self, name, fake_fps=0):
        if name not in self.data_sets:
            self._load_data_set(name, fake_fps)
        return self.data_sets[name]

    def get_data_collection(self, name, fake_fps=0):
        if name not in self.data_collections:
            self._load_data_collection(name, fake_fps)
        return self.data_collections[name]

    def get_sample(self, set_name, sample_name, fake_fps=0):
        ds = self.get_data_set(set_name, fake_fps)
        sample = ds.samples_by_name[sample_name]
        return sample

    def get_ros_sample(self, node_id):
        return LiveSample(node_id)

    def evaluate_sample_list(self, sample_names, tracking_conf):
        fake_fps = 0
        if 'fake_fps' in tracking_conf:
            fake_fps = tracking_conf['fake_fps']
        samples = []
        for sname in sample_names:
            p1, p2 = sname.split('/', 1)
            if p1 == 'SET':
                # this is a full sample set:
                set_name = p2
                ds = self.get_data_set(set_name, fake_fps)
                samples.extend(ds.samples)
            elif p1 == 'COLLECTION':
                # this is a collection of samples
                collection_name = p2
                dc = self.get_data_collection(collection_name, fake_fps)
                dc.load_samples()
                samples.extend(dc.samples)
            else:
                # this is a single sample:
                if p1 == 'ros':
                    samples.append(self.get_ros_sample(p2))
                else:
                    set_name, sample_name = p1, p2
                    samples.append(self.get_sample(set_name, sample_name, fake_fps))
        return samples

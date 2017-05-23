"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import os
import zipfile
import logging
import re
import io

import yaml
from PIL import Image

from hiob.rect import Rect

logger = logging.getLogger(__name__)


class DataSetException(Exception):
    pass


class DataDirectory(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        conf_dir = 'conf'
        self.data_set_dir = os.path.join(conf_dir, 'data_sets')
        self.data_collection_dir = os.path.join(conf_dir, 'collections')

        self.data_sets = {}
        self.data_collections = {}

    def _load_data_set(self, name):
        set_file = os.path.join(self.data_set_dir, name + '.yaml')
        ds = DataSet(name, self.data_dir)
        with open(set_file, "r") as f:
            y = yaml.safe_load(f)
            ds.load(y)
        self.data_sets[name] = ds

    def _load_data_collection(self, name):
        col_file = os.path.join(self.data_collection_dir, name + '.yaml')
        dc = DataCollection(self, name)
        with open(col_file, "r") as f:
            y = yaml.safe_load(f)
            dc.load(y)
        self.data_collections[name] = dc

    def get_data_set(self, name):
        if name not in self.data_sets:
            self._load_data_set(name)
        return self.data_sets[name]

    def get_data_collection(self, name):
        if name not in self.data_collections:
            self._load_data_collection(name)
        return self.data_collections[name]

    def get_sample(self, set_name, sample_name):
        ds = self.get_data_set(set_name)
        sample = ds.samples_by_name[sample_name]
        return sample

    def evaluate_sample_list(self, sample_names):
        samples = []
        for sname in sample_names:
            p1, p2 = sname.split('/', 1)
            if p1 == 'SET':
                # this is a full sample set:
                set_name = p2
                ds = self.get_data_set(set_name)
                samples.extend(ds.samples)
            elif p1 == 'COLLECTION':
                # this is a collection of samples
                collection_name = p2
                dc = self.get_data_collection(collection_name)
                dc.load_samples()
                samples.extend(dc.samples)
            else:
                # this is a single sample:
                set_name, sample_name = p1, p2
                samples.append(self.get_sample(set_name, sample_name))
        return samples


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

    def load_samples(self):
        if self.loaded:
            return
        for p1, p2 in self.samples_parsed:
            if p1 == 'SET':
                # this is a full data set
                ds = self.directory.get_data_set(p2)
                self.samples.extend(ds.samples)
            elif p2 == 'COLLECTION':
                # this is a collection within a collection - not supported
                raise NotImplementedError(
                    "Collections within collections of samples are not supported.")
            else:
                set_name, sample_name = p1, p2
                self.samples.append(
                    self.directory.get_sample(set_name, sample_name))
        self.loaded = True


class DataSet(object):

    def __init__(self, name, data_dir):
        self.name = name
        self.description = None
        self.samples = []
        self.samples_by_name = {}
        self.total_samples = 0
        self.format = None
        self.path = os.path.join(data_dir, name)

    def load(self, definition):
        if 'description' in definition:
            self.description = definition['description']
        if 'format' in definition:
            self.format = definition['format']
        for sdef in definition['samples']:
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


class Sample(object):

    def __init__(self, data_set, name):
        self.data_set = data_set
        self.name = name
        self.set_name = data_set.name
        self.full_name = self.set_name + '/' + self.name
        self.loaded = False
        self.attributes = []
        self.first_frame = None
        self.last_frame = None
        self.total_frames = None
        self.actual_frames = None
        # data from actual data set sample:
        self.images = []
        self.ground_truth = []

    def __repr__(self):
        return '<Sample {name}/{set_name}>'.format(name=self.name, set_name=self.set_name)

    def load_tb100zip(self):
        if '.' in self.name:
            # is sample like 'Jogging.2' with multiple ground truths
            name, number = self.name.split('.')
            gt_name = name + '/groundtruth_rect.{}.txt'.format(number)
        else:
            name = self.name
            number = None
            gt_name = name + '/groundtruth_rect.txt'

        path = os.path.join(self.data_set.path, name + '.zip')
        with zipfile.ZipFile(path, 'r') as zf:
            # get ground truth:
            r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
            gt = []
            with zf.open(gt_name, 'r') as gtf:
                for line in gtf.readlines():
                    if type(line) is bytes:
                        line = line.decode('utf-8')
                    m = r.match(line)
                    if m:
                        gt.append(Rect(m.groups()))
                    else:
                        gt.append(None)
            # get frames (images):
            img_paths = []
            r = re.compile(r".+/img/\d+\.[0-9a-zA-Z]+")
            for fn in zf.namelist():
                if r.match(fn):
                    img_paths.append(fn)
            img_paths.sort()
            images = []
            for img_path in img_paths:
                data = zf.read(img_path)
                stream = io.BytesIO(data)
                im = Image.open(stream)
                if im.mode != "RGB":
                    # convert s/w to colour:
                    im = im.convert("RGB")
                images.append(im)

            # fix weird frame cases like David and Football1
            if self.first_frame is None:
                self.first_frame = 1
            if self.last_frame is None:
                self.last_frame = len(images) + 1
            self.frame_offset = self.first_frame - 1
            if self.first_frame != 1:
                # truncate everything before first frame:
                gt = gt[self.frame_offset:]
                images = images[self.frame_offset:]
            if len(images) < len(gt):
                raise DataSetException(
                    "More ground truth frames than images in DataSet {}/{}".format(self.set_name, self.name))
            if len(images) > len(gt):
                # some samples have more images than gt at the end, cut them:
                images = images[:len(gt)]
        # save what we got
        self.ground_truth = gt
        self.images = images
        # first ground truth position is initial position
        self.initial_position = gt[0]
        # verify number of actual frames from data set file:
        if len(images) != self.actual_frames:
            logger.error(
                "Wrong number for actual_images in sample {}".format(self))
            self.actual_frames = len(images)

    def load_princeton(self):
        # find out, of sample is Evaluation or Validation set:
        dir_e = os.path.join(self.data_set.path, 'EvaluationSet', self.name)
        dir_v = os.path.join(self.data_set.path, 'ValidationSet', self.name)
        dir_path = None
        if os.path.isdir(dir_e):
            dir_path = dir_e
        elif os.path.isdir(dir_v):
            dir_path = dir_v
        else:
            raise DataSetException(
                'Could find princeton sample {} neither in EvaluationSet nor ValidationSet'.format(self.name))
        # read initial position (position in first frame):
        init_path = os.path.join(dir_path, 'init.txt')
        with open(init_path, "r") as f:
            line = f.readline().strip()
        self.initial_position = Rect(line.split(','))
        # read images:
        img_dir = os.path.join(dir_path, "rgb")
        r = re.compile(r"^\w+-(\d+)-(\d+)\.\w+$")
        files = {}
        for fname in os.listdir(img_dir):
            # eg "r-266705-6.png" for 6th frame...
            m = r.match(fname)
            # print(fname, m)
            if m:
                # found one
                num = int(m.group(2))
                files[num] = fname
        # collected all files, now loading them, sortedly:
        images = []
#        print(sorted(files))
        for key in sorted(files):
            fname = files[key]
            fpath = os.path.join(img_dir, fname)
            im = Image.open(fpath).copy()
            if im.mode != "RGB":
                # convert s/w to colour:
                im = im.convert("RGB")
            images.append(im)
        self.images = images
        # verify number of actual frames from data set file:
        if len(images) != self.actual_frames:
            logger.error(
                "Wrong number for actual_images in sample {}".format(self))
            self.actual_frames = len(images)
        # gt:
        gt_path = init_path = os.path.join(dir_path, self.name + '.txt')
        if os.path.isfile(gt_path):
            # gt exists:
            gts = []
            r = re.compile(r'^(\d+),(\d+),(\d+),(\d+)')
            with open(gt_path) as f:
                for line in f.readlines():
                    if line.startswith('NaN'):
                        # no rect for this frame
                        gts.append(None)
                    else:
                        m = r.match(line)
                        if m:
                            gts.append(Rect(m.groups()))
                        else:
                            logger.error(
                                "Invalid line in ground truth: '%s'", line.strip())
            self.ground_truth = gts
        logger.info("loaded %s with %d frames", self.name, len(images))

    def load(self):
        if self.loaded:
            return
        if self.data_set.format == 'tb100zip':
            self.load_tb100zip()
        elif self.data_set.format == 'princeton':
            self.load_princeton()
        else:
            raise DataSetException("Unknown DataSet format:" + self.set.format)
        self.loaded = True

    def unload(self):
        """
        Free most data.

        Delete the images in the sample to free most of the memory taken by sample.
        """
        if self.loaded:
            self.images = []
            self.loaded = False

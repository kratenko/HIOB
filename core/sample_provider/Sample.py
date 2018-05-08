"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import os
import zipfile
import logging
import re
import io
import tensorflow as tf
import numpy as np
import time

from PIL import Image

from ..Rect import Rect

from .DataSetException import DataSetException

logger = logging.getLogger(__name__)


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
        self.image_cache = []
        self.img_paths = []
        self.ground_truth = []
        self.current_frame_id = -1
        self.initial_position = None
        self.frame_offset = None
        self.capture_size = None

    def __repr__(self):
        return '<Sample {name}/{set_name}>'.format(name=self.name, set_name=self.set_name)

    def load_tb100zip(self, log_context=None):

        msg = "Start loading sample {}".format(self.name)
        if log_context is not None:
            with log_context(logger):
                logger.info(msg)
        else:
            logger.info(msg)
        if '.' in self.name:
            # is sample like 'Jogging.2' with multiple ground truths
            name, number = self.name.split('.')
            gt_name = 'groundtruth_rect.{}.txt'.format(number)
        else:
            name = self.name
            gt_name = 'groundtruth_rect.txt'

        path = os.path.join(self.data_set.path, name + '.zip')
        self.zip_file = zipfile.ZipFile(path, 'r')
        # get ground truth:
        r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
        gt = []
        with self.zip_file.open(gt_name, 'r') as gtf:
            for line in gtf.readlines():
                if type(line) is bytes:
                    line = line.decode('utf-8')
                m = r.match(line)
                if m:
                    gt.append(Rect(m.groups()))
                else:
                    gt.append(None)
        # get frames (images):
        self.img_paths = []
        r = re.compile(r"img/\d+\.[0-9a-zA-Z]+")
        for fn in self.zip_file.namelist():
            if r.match(fn):
                self.img_paths.append(fn)
        self.img_paths.sort()
        images = []
        # fix weird frame cases like David and Football1
        if self.first_frame is None:
            self.first_frame = 1
        self.frame_offset = self.first_frame - 1

        c = 1
        for img_path in self.img_paths[self.frame_offset:]:
            c += 1
            im = self.load_tb100_image(img_path)
            images.append(im)

        self.capture_size = tuple(reversed(images[0].shape[:-1]))

        if self.last_frame is None:
            self.last_frame = len(self.img_paths) + 1
        if self.first_frame != 1:
            # truncate everything before first frame:
            gt = gt[self.frame_offset:]
            #images = images[self.frame_offset:]
        if len(images) < len(gt):
            # raise DataSetException(
            #     "More ground truth frames than images in DataSet {}/{}".format(self.set_name, self.name))
            gt = gt[:len(images) - 1]
            if len(images) < len(gt):
                raise Exception("still more gts")

        if len(images) > len(gt):
            # some samples have more images than gt at the end, cut them:
            images = images[:len(gt)]
        # save what we got
        self.ground_truth = gt
        self.image_cache = images
        # first ground truth position is initial position
        self.initial_position = gt[0]
        # verify number of actual frames from data set file:
        if len(images) != self.actual_frames:
            logger.error(
                "Wrong number for actual_images in sample {}".format(self))
            # self.actual_frames = len(images)

        msg = "Sample '{}' loaded".format(self.name)
        if log_context is not None:
            with log_context(logger):
                logger.info(msg)
        else:
            logger.info(msg)


    def load_tb100_image(self, img_path):
        data = self.zip_file.read(img_path)
        stream = io.BytesIO(data)
        im = np.array(Image.open(stream))
        return im

    def load_princeton(self, log_context=None):
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
                'Could find princeton sample {} neither in EvaluationSet nor ValidationSet'.format(self.data_set.path + "/" + self.name))
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
        self.image_cache = images
        # verify number of actual frames from data set file:
        if len(images) != self.actual_frames:
            logger.error(
                "Wrong number for actual_images in sample {}".format(self))
            #self.actual_frames = len(images)
        # gt:
        gt_path = os.path.join(dir_path, self.name + '.txt')
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

        msg = "loaded %s with %d frames", self.name, len(images)
        if log_context is not None:
            with log_context(logger):
                logger.info(msg)
        else:
            logger.info(msg)

    def load(self, log_context=None):
        if self.loaded:
            return
        if self.data_set.format == 'tb100zip':
            self.load_tb100zip(log_context)
        elif self.data_set.format == 'princeton':
            self.load_princeton(log_context)
        else:
            raise DataSetException("Unknown DataSet format:" + self.set.format)
        self.loaded = True
        self.total_frames = self.actual_frames

    def unload(self):
        """
        Free most data.

        Delete the images in the sample to free most of the memory taken by sample.
        """
        if self.loaded:
            self.image_cache = []
            self.loaded = False

    def get_image(self, img_id):
        if img_id >= len(self.img_paths):
            return None
        #if len(self.image_cache) <= img_id or self.image_cache[img_id] is None:
        #    self.image_cache[-9] = None
        #    print("img_id: {}".format(img_id))
        #    print("cache_length: {}".format(len(self.image_cache)))
        #    self.image_cache.append(self.load_tb100_image(self.img_paths[img_id]))
        return self.image_cache[img_id]

    def get_ground_truth(self, gt_id):
        if len(self.ground_truth) > gt_id and len(self.ground_truth) > self.actual_frames:
            return self.ground_truth[gt_id]
        else:
            return None

    async def get_next_frame_data(self):
        self.current_frame_id += 1
        return [
            self.get_image(self.current_frame_id),
            self.get_ground_truth(self.current_frame_id)]

    def frames_left(self):
        return max(0, min(self.actual_frames, len(self.img_paths)) - self.current_frame_id - 1)

    def count_frames_processed(self):
        return self.current_frame_id + 1

    def count_frames_skipped(self):
        return 0

    def get_actual_frames(self):
        return self.actual_frames
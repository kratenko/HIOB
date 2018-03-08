"""
Created on 2016-12-08

@author: Peer Springstübe
"""

import os
import zipfile
import logging
import re
import io

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
        self.images = []
        self.ground_truth = []
        self.current_frame_id = -1
        self.initial_position = None
        self.frame_offset = None

    def __repr__(self):
        return '<Sample {name}/{set_name}>'.format(name=self.name, set_name=self.set_name)

    def load_tb100zip(self):
        if '.' in self.name:
            # is sample like 'Jogging.2' with multiple ground truths
            name, number = self.name.split('.')
            gt_name = name + '/groundtruth_rect.{}.txt'.format(number)
        else:
            name = self.name
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
        self.images = images
        # verify number of actual frames from data set file:
        if len(images) != self.actual_frames:
            logger.error(
                "Wrong number for actual_images in sample {}".format(self))
            self.actual_frames = len(images)
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
        self.total_frames = self.actual_frames

    def unload(self):
        """
        Free most data.

        Delete the images in the sample to free most of the memory taken by sample.
        """
        if self.loaded:
            self.images = []
            self.loaded = False

    def get_image(self, img_id):
        return self.images[img_id]

    def get_ground_truth(self, gt_id):
        if len(self.ground_truth) > gt_id:
            return self.ground_truth[gt_id]
        else:
            return None

    async def get_next_frame_data(self):
        self.current_frame_id += 1
        return [
            self.get_image(self.current_frame_id),
            self.get_ground_truth(self.current_frame_id)]

    def frames_left(self):
        return max(0, len(self.images) - self.current_frame_id - 1)

    def count_frames_processed(self):
        return self.current_frame_id + 1

    def count_frames_skipped(self):
        return 0

    def get_actual_frames(self):
        return self.actual_frames
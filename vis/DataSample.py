import os
import io
import zipfile
import re
from PIL import Image

import logging
import scipy.io

logger = logging.getLogger(__name__)


def loc2affgeo(location, particle_size=64):
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    gw = w / particle_size
    gh = h / w
    geo = [cx, cy, gw, gh]
    return geo


def affgeo2loc(geo, particle_size=64):
    cx, cy, pw, ph = geo
    w = pw * particle_size
    h = ph * w
    x = cx - (w - 1) / 2
    y = cy - (h - 1) / 2
    return [x, y, w, h]


class DataSample(object):

    def __init__(self):
        self.fcnt_position = None
        pass

    def _load_image_from_zip(self, zf, path):
        data = zf.read(path)
        stream = io.BytesIO(data)
        im = Image.open(stream)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im

    def _load_image_from_dir(self, path):
        with Image.open(path) as im:
            if im.mode != "RGB":
                return im.convert("RGB")
            else:
                return im.copy()

    def _extract_name_from_zip(self, zf):
        for fn in zf.namelist():
            if fn.endswith('/img/'):
                return fn[:-5]
        return None

    def _extract_image_paths_from_zip(self, zf):
        paths = []
        r = re.compile(r".+/img/\d+\.[0-9a-zA-Z]+")
        for fn in zf.namelist():
            if r.match(fn):
                paths.append(fn)
        return sorted(paths)

    def _extract_groundtruth_from_zip(self, zf):
        paths = []
        r = re.compile(r".+/groundtruth_rect.*\.txt")
        for fn in zf.namelist():
            if r.match(fn):
                paths.append(fn)
        return sorted(paths)

    def _groundtruth_line(self, line):
        r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
        m = r.match(str(line))
        if m:
            return tuple(int(m.group(n)) for n in (1, 2, 3, 4))
        else:
            return None

    def _load_groundtruth_from_zip(self, zf, path):
        rects = []
        with zf.open(path, 'r') as tf:
            for line in tf:
                rects.append(self._groundtruth_line(line))
        return rects

    def _load_groundtruth_from_dir(self, path):
        rects = []
        with open(path, 'r') as tf:
            for line in tf:
                rects.append(self._groundtruth_line(line))
        return rects

    def load_from_zip(self, path):
        self.path = path
        with zipfile.ZipFile(path, 'r') as zf:
            # extract info from filenames within zip:
            self.name = self._extract_name_from_zip(zf)
            self.image_paths = self._extract_image_paths_from_zip(zf)
            self.groundtruth_paths = self._extract_groundtruth_from_zip(zf)
            # load images:
            self.frames = [self._load_image_from_zip(
                zf, image_path) for image_path in self.image_paths]
            # load truths
            self.groundtruths = [self._load_groundtruth_from_zip(
                zf, truth_path) for truth_path in self.groundtruth_paths]

    def load_from_dir(self, path):
        self.path = path
        self.name = os.path.basename(path)
        filelist = os.listdir(path)

        gtnames = [x for x in filelist if re.match(
            r"^groundtruth.+\.txt$", x, re.IGNORECASE)]
        self.groundtruth_paths = sorted(
            [os.path.join(path, x) for x in gtnames])

        imgpath = os.path.join(path, 'img')
        imgnames = [x for x in os.listdir(imgpath) if re.match(
            r"^\d+\.jpe?g$", x, re.IGNORECASE)]
        self.image_paths = sorted([os.path.join(imgpath, x) for x in imgnames])
        self.frames = [self._load_image_from_dir(p) for p in self.image_paths]
        self.groundtruths = [self._load_groundtruth_from_dir(
            truth_path) for truth_path in self.groundtruth_paths]
        if 'position.mat' in filelist:
            pos_path = os.path.join(self.path, 'position.mat')
            mpos = scipy.io.loadmat(pos_path)['position']
            self.fcnt_position = []
            for i in range(len(mpos[0])):
                geo = mpos[0][i], mpos[1][i], mpos[2][i], mpos[4][i]
                loc = affgeo2loc(geo)
                self.fcnt_position.append(loc)
    # --

    def _load_princeton_image(self, path):
        with open(path) as image_file:
            im = Image.open(image_file)
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
        return im

    def _load_princeton_gt(self, path):
        _, name = os.path.split(path)
        gt_path = os.path.join(path, name + ".txt")
        gt = []
        if os.path.isfile(gt_path):
            gt = []
            with open(gt_path) as input_file:
                for line in input_file:
                    (x, y, w, h, _) = line.split(",")
                    if x == 'NaN':
                        gt.append(None)
                    else:
                        gt.append([int(x), int(y), int(w), int(h)])
        return gt

    def load_princeton(self, path):
        self.path = path
        image_dir = os.path.join(path, 'rgb')
        image_paths = os.listdir(image_dir)
        r = re.compile(r"^r-(\d+)-(\d+).(\w+)$")
        d = {}
        for p in image_paths:
            m = r.match(p)
            if m:
                (_, num, _) = m.groups()
                d[int(num)] = p
        file_names = [d[k] for k in sorted(d.keys())]
        self.frames = [self._load_princeton_image(
            os.path.join(image_dir, file_name)) for file_name in file_names]
        self.groundtruths = [self._load_princeton_gt(path)]

    # ---

    def _determine_type(self, path):
        logger.info("try do determine data set type for '{}'".format(path))
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path)
        is_file = os.path.isfile(path)
        if not exists:
            raise FileNotFoundError(
                "No such file or directory: '{}'".format(path))
        if is_dir:
            filenames = os.listdir(path)
            if all(name in filenames for name in ['img', 'cfg.mat']):
                # check if img-dir and groundtruth exist:
                img_dir = os.path.join(path, 'img')
                gt0 = os.path.join(path, 'groundtruth_rect.txt')
                gt1 = os.path.join(path, 'groundtruth_rect.1.txt')
                gt2 = os.path.join(path, 'groundtruth_rect.2.txt')
                has_img_dir = os.path.isdir(img_dir)
                has_gt0 = os.path.isfile(gt0)
                has_gt1 = os.path.isfile(gt1)
                has_gt2 = os.path.isfile(gt2)
                matcher = re.compile(r"^\d+\.jpg$")

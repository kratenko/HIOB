"""
Created on 2016-11-01

@author: Peer Springstübe
"""

import os
import logging
import re
from PIL import Image
from hiob.rect import Rect
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
    set_name = None
    sample_name = None
    images = None
    ground_truth = None
    pre_gid = None
    pre_lid = None
    pre_position = None

    def __init__(self):
        pass

    def full_name(self):
        return self.set_name + "/" + self.sample_name


def _load_tb100_images(path):
    # in tb100 images are stored as nnnn.jpg in subdirectory img
    # this just expects files to be in alphanumeric order, but they need not
    # start with 0001.jpg
    fname_pattern = re.compile(r"^\d+\.(jpe?g|png)$", re.IGNORECASE)
    img_path = os.path.join(path, 'img')
    logger.debug("Loading image files from directory '%s'", img_path)
    file_names = os.listdir(path=img_path)
    images = []
    for fname in sorted(file_names):
        if fname_pattern.match(fname):
            # filename matches, load file
            fpath = os.path.join(img_path, fname)
            logger.debug("Loading image file `%s`", fpath)
            with open(fpath, mode="rb") as f:
                im = Image.open(f, mode='r')
                # make sure image is really loaded to memory:
                im.load()
                # ensure format is RGB
                if im.mode != "RGB":
                    im = im.convert("RGB")
                images.append(im)
    logger.debug("Loaded %d image files", len(images))
    return images


def _load_tb100_gt(path):
    r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
    rects = []
    with open(path, 'r') as tf:
        for line in tf:
            m = r.match(line)
            if m:
                rect = Rect(tuple(int(m.group(n)) for n in (1, 2, 3, 4)))
            else:
                rect = None
            rects.append(rect)
    return rects


def load_tb100_sample(name, set_path=None):
    logger.info("Loading sample '%s' from data set 'tb100'", name)
    # generate paths needed for loading:
    if set_path is None:
        set_path = os.path.join("/data", "Peer", "data", "tb100_unzipped")
    parts = name.split(".")
    if len(parts) == 1:
        # e.g.: Tiger1
        sample_path = os.path.join(set_path, name)
        gt_path = os.path.join(sample_path, "groundtruth_rect.txt")
        gid_path = os.path.join(sample_path, 'gid.mat')
        lid_path = os.path.join(sample_path, 'lid.mat')
        position_path = os.path.join(sample_path, 'position.mat')
    elif len(parts) == 2:
        # e.g.: Jogging.1
        sample_path = os.path.join(set_path, parts[0])
        gt_path = os.path.join(
            sample_path, "groundtruth_rect." + parts[1] + ".txt")
        gid_path = os.path.join(sample_path, "gid." + parts[1] + ".mat")
        lid_path = os.path.join(sample_path, "lid." + parts[1] + ".mat")
        position_path = os.path.join(
            sample_path, "position." + parts[1] + ".mat")
    else:
        logger.fatal("Invalid sample name for tb100: '%s'", name)
    frames = _load_tb100_images(sample_path)
    gt = _load_tb100_gt(gt_path)
    ds = DataSample()
    ds.images = frames
    ds.ground_truth = gt
    ds.sample_name = name
    ds.set_name = "tb100"
    if os.path.exists(gid_path):
        ds.pre_gid = scipy.io.loadmat(gid_path)['gid']
    if os.path.exists(lid_path):
        ds.pre_lid = scipy.io.loadmat(lid_path)['lid']
    if os.path.exists(position_path):
        #ds.pre_position = scipy.io.loadmat(position_path)['position']
        ds.pre_position = []
        mpos = scipy.io.loadmat(position_path)['position']
        for i in range(len(mpos[0])):
            geo = mpos[0][i], mpos[1][i], mpos[2][i], mpos[4][i]
            loc = affgeo2loc(geo)
            ds.pre_position.append(Rect(loc))
    return ds


def load_sample(data_set, name, config=None):
    if data_set == 'tb100':
        if config:
            path = config['data_sets']['tb100']['path']
        else:
            path = None
        return load_tb100_sample(name, path)
    else:
        logger.fatal("Unknown data set: %s", data_set)

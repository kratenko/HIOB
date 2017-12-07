"""
Created on 2016-11-29

@author: Peer Springstübe
"""
import numpy as np
import scipy.ndimage
import tensorflow as tf
from matplotlib import pyplot as plt

from gauss import gen_gauss_mask
from .util import loc2xgeo, xgeo2loc
from .Pursuer import Pursuer
from Rect import Rect
from concurrent import futures
from functools import partial
import time
import multiprocessing


import logging

logger = logging.getLogger(__name__)


def position_quality_helper(img_mask, roi, total_max, mask_sum, location):
    return SwarmPursuer.position_quality(img_mask, Rect(location), roi, mask_sum) / total_max


class SwarmPursuer(Pursuer):

    def __init__(self):
        self.dtype = tf.float32
        workers = multiprocessing.cpu_count()
        self.thread_executor = futures.ThreadPoolExecutor(max_workers=workers)

    def configure(self, configuration):
        self.configuration = configuration
        pconf = configuration['pursuer']
        self.particle_count = pconf['particle_count']
        self.target_lower_limit = float(pconf['target_lower_limit'])
        self.target_punish_low = float(pconf['target_punish_low'])
        self.target_punish_outside = float(pconf['target_punish_outside'])
        self.np_random = configuration['np_random']

    def setup(self, session):
        self.session = session

    def generate_geo_particles(self, geo, img_size, lost):
        if lost == 0:
            num_particles = self.particle_count
            #spread = 0.4
            spread = 1.0
        elif lost == 1:
            num_particles = self.particle_count * 2
            # spread = 2.0
            spread = 5.0
        else:
            raise ValueError("Invalid value for lost: {}".format(lost))

        # geo = loc2affgeo(loc)
        geos = np.tile(geo, (num_particles, 1)).T
#        r = self.np_random.randn(4, num_particles)
#        r *= 0.2
        r1 = self.np_random.randn(2, num_particles) * spread
        r2 = self.np_random.randint(-1, 2, (2, num_particles))
        r = np.concatenate((r1, r2))
        #f = np.tile([10, 10, .01, .01], (num_particles, 1)).T
#        f = np.tile([10, 10, 0.004, 0], (num_particles, 1)).T
        f = np.tile([10, 10, 0, 0], (num_particles, 1)).T
        rn = np.multiply(r, f)

        geos += rn
        #
        if False:
            geos[2, geos[2, :] < 0.05] = 0.05
            geos[2, geos[2, :] > 0.95] = 0.95
            geos[3, geos[3, :] < 0.10] = 0.10
            geos[3, geos[3, :] > 10.0] = 10.0
            w = img_size[0]
            h = img_size[1]
            geos[0, geos[0, :] < (0.05 * w)] = 0.05 * w
            geos[0, geos[0, :] > (0.95 * w)] = 0.95 * w
            geos[1, geos[1, :] < (0.05 * h)] = 0.05 * h
            geos[1, geos[1, :] > (0.95 * h)] = 0.95 * h

        return (geos + rn).T

    def generate_particles(self, loc, img_size, lost):
        geo = loc2xgeo(loc)
        geos = self.generate_geo_particles(geo, img_size, lost)
        locs = [xgeo2loc(g) for g in geos]
        # add previous position, to make sure there is at least one valid
        # position:
        locs.append(loc)
        return locs

    def upscale_mask(self, mask, roi, image_size):
        # scale prediction mask up to size of roi (not of sroi!):
        relation = roi.width / self.mask_size[0], \
            roi.height / self.mask_size[1]
        roi_mask = scipy.ndimage.zoom(mask.reshape(self.mask_size), relation)
        # crop low values
        roi_mask[roi_mask < self.target_lower_limit] = self.target_punish_low
        # put mask in capture image mask:
        img_mask = np.full(image_size, self.target_punish_outside)
        img_mask[int(roi.top): int(roi.bottom),
                 int(roi.left): int(roi.right)] = roi_mask
        return img_mask

    @staticmethod
    def position_quality(image_mask, pos, roi, image_mask_sum):
        #logger.info("QUALI: %s, %s", image_mask.shape, pos)
        # too small?
        if pos.width < 8 or pos.height < 8:
            return -1e12
        # outside roi?
        if pos.left < roi.left or pos.top < roi.top or pos.right > roi.right or pos.bottom > roi.bottom:
            return -1e12
        #p1 = time.time()
        inner = (image_mask[
            int(pos.top):int(pos.bottom - 1),
            int(pos.left):int(pos.right - 1)]).sum()
        #p2 = time.time()
        inner_fill = inner / pos.pixel_count()
        # return inner_fill * inner_part
        #p3 = time.time()
        outer = image_mask_sum - inner
        #p4 = time.time()
        outer_fill = outer / max(roi.pixel_count() - pos.pixel_count(), 1)
        #p5 = time.time()

        """total = p5 - p1
        t1 = p2 - p1
        t2 = p3 - p2
        t3 = p4 - p3
        t4 = p5 - p4

        print("image_mask dim: {}".format(str(image_mask.shape)))
        print("inner dim: {}".format(str(image_mask[
                                         int(pos.top):int(pos.bottom - 1),
                                         int(pos.left):int(pos.right - 1)].shape)))
        print("part1: {} ({}%); part2: {} ({}%); part3: {} ({}%); part4: {} ({}%)".format(t1, t1 / total, t2,
                                                                                          t2 / total, t3, t3 / total,
                                                                                          t4, t4 / total))"""
        return max(inner_fill - outer_fill, 0.0)

    def pursue(self, state, frame, lost=0):
        logger.info("Predicting position for frame %s, Lost: %d", frame, lost)
        # TODO: not here...
        self.mask_size = self.configuration['mask_size']
        #
        mask = frame.prediction_mask.copy()

        mask[mask < self.target_lower_limit] = self.target_punish_low
        mask[mask < 0.0] = 0.0

        #print("a", mask.max(), mask.min(), np.average(mask))
        img_size = [frame.capture_image.size[1], frame.capture_image.size[0]]
        img_mask = self.upscale_mask(mask, frame.roi, img_size)
        #print("a", img_mask.max(), img_mask.min(), np.average(img_mask))
        frame.image_mask = img_mask
        locs = self.generate_particles(
            frame.previous_position, frame.capture_image.size, lost)
        #total = np.sum(img_mask)
        #total_max = np.sum(img_mask[img_mask > 0])
#        total_max = np.sum(np.abs(img_mask))

        img_mask_sum = img_mask.sum()
        total_max = 1
        func = partial(position_quality_helper, img_mask, frame.roi, total_max, img_mask_sum)
        quals = list(self.thread_executor.map(func, locs))
        #quals = []
        # for n, l in enumerate(locs):
        #    r = Rect(l)
        #    q = self.position_quality(img_mask, r)
        #    logger.info("%d, %r: %f", n, r, q)
        #    quals.append(q)
        best_arg = np.argmax(quals)
        frame.predicted_position = Rect(locs[best_arg])
        # quality of prediction needs to be absolute, so we normalise it with
        # the "perfect" value this prediction would have:
        perfect_quality = 1
        #print(quals[best_arg], perfect_quality)
        frame.prediction_quality = max(
            0.0, min(1.0, quals[best_arg] / perfect_quality))
        logger.info("Prediction: %s, quality: %f",
                    frame.predicted_position, frame.prediction_quality)
        return frame.predicted_position
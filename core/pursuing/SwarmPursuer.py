"""
Created on 2016-11-29

@author: Peer Springstübe
"""
import numpy as np
import scipy.ndimage
import tensorflow as tf
from functools import partial
from matplotlib import pyplot as plt

from .util import loc2xgeo, xgeo2loc
from .Pursuer import Pursuer
from ..Rect import Rect
from concurrent import futures
import time
import multiprocessing

avgs = []

import logging

logger = logging.getLogger(__name__)


class SwarmPursuer(Pursuer):

    def __init__(self):
        self.dtype = tf.float32
        #self.thread_executor = futures.ProcessPoolExecutor(max_workers=workers)
        self.thread_executor = None

    def configure(self, configuration):
        self.configuration = configuration
        pconf = configuration['pursuer']
        self.particle_count = pconf['particle_count']
        self.particle_scale_factor = pconf['particle_scale_factor'] if 'particle_scale_factor' in pconf else 1.0
        self.target_lower_limit = float(pconf['target_lower_limit'])
        self.target_punish_low = float(pconf['target_punish_low'])
        self.target_punish_outside = float(pconf['target_punish_outside'])
        available_cpus = multiprocessing.cpu_count()
        self.worker_count = min(configuration['max_cpus'], available_cpus * 4) if 'max_cpus' in configuration else available_cpus
        print("Spawning {} workers.".format(self.worker_count));
        self.thread_executor = futures.ThreadPoolExecutor(max_workers=self.worker_count)
        self.np_random = configuration['np_random']

    def set_initial_position(self, pos):
        self.initial_location = pos

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

        # geos += rn
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
        locs = [Rect(xgeo2loc(g)) for g in geos]
        # add previous position, to make sure there is at least one valid
        # position:
        locs.append(loc)
        return locs

    def upscale_mask(self, mask, roi, image_size):
        # scale prediction mask up to size of roi (not of sroi!):
        relation = roi.width / self.mask_size[0], roi.height / self.mask_size[1]
        #roi_mask = scipy.ndimage.zoom(mask.reshape(self.mask_size), relation)
        roi_mask = mask.reshape((mask.shape[1], mask.shape[2]))
        # crop low values
        roi_mask[roi_mask < self.target_lower_limit] = self.target_punish_low
        # put mask in capture image mask:
        img_mask = np.full((round(image_size[0] / relation[0]),
                            round(image_size[1] / relation[1])), self.target_punish_outside)
        img_mask[round(roi.top / relation[1]): round(roi.bottom / relation[1]),
                 round(roi.left / relation[0]): round(roi.right / relation[0])] = roi_mask
        return img_mask, relation

    def position_quality(self, pos, roi, image_mask_sum, inner_sum, scale_factor):
        #logger.info("QUALI: %s, %s", image_mask.shape, pos)
        # too small?
        # p1 = time.time()
        scale_factor_squared = scale_factor[0] *scale_factor[1]
        if pos.width < 8 or pos.height < 8:
            return -1e12
        # outside roi?
        if pos.left < roi.left or pos.top < roi.top or pos.right > roi.right or pos.bottom > roi.bottom:
            return -1e12
        # p2 = time.time()
        """inner = img_mask[
                int(pos.top):int(pos.bottom - 1),
                int(pos.left):int(pos.right - 1)].sum()"""
        inner = inner_sum

        # p3 = time.time()
        inner_fill = inner / (pos.pixel_count() / scale_factor_squared)
        outer = image_mask_sum - inner

        outer_fill = outer / max((roi.pixel_count() - pos.pixel_count()) / scale_factor_squared, 1)
        # p4 = time.time()
        # p5 = time.time()

        """total = p4 - p1
        t1 = p2 - p1
        t2 = p3 - p2
        t3 = p4 - p3
        # t4 = p5 - p4

        print("part1: {:.2} ({:.2}); part2: {:.2} ({:.2}); part3: {:.2} ({:.2});".format(t1, t1 / total, t2,
                                                                                         t2 / total, t3, t3 / total))"""

        # for dynamic rescaling, pass 'punish_low=True' to calculate_sum and use 'inner' as quality
        if self.particle_scale_factor == 1.0: # no scaling
            quality = max(inner_fill - outer_fill, 0.0)
        else:
            quality = max(inner, 0.0)
        # print("quality: {}".format(quality))
        return quality

    def pursue(self, state, frame, lost=0):
        ps = [time.time()]  # 0
        logger.info("Predicting position for frame %s, Lost: %d", frame, lost)
        # TODO: not here...
        self.mask_size = self.configuration['mask_size']
        #
        mask = frame.prediction_mask.copy()

        ps.append(time.time())  # 1

        mask[mask < self.target_lower_limit] = self.target_punish_low
        mask[mask < 0.0] = 0.0

        #print("a", mask.max(), mask.min(), np.average(mask))
        img_size = [frame.size[1], frame.size[0]]

        ps.append(time.time())  # 2
        img_mask, scale_factor = self.upscale_mask(mask, frame.roi, img_size)
        #print("a", img_mask.max(), img_mask.min(), np.average(img_mask))
        frame.image_mask = img_mask

        ps.append(time.time())  # 3

        locs = self.generate_particles(
            frame.previous_position, frame.size, lost)
        #total = np.sum(img_mask)
        #total_max = np.sum(img_mask[img_mask > 0])
#        total_max = np.sum(np.abs(img_mask))

        ps.append(time.time())  # 4
        img_mask_sum = img_mask.sum()
        ps.append(time.time())  # 5
        total_max = 1
        """func = partial(position_quality_helper, img_mask, frame.roi, total_max, img_mask_sum)
        p4 = time.time()
        quals = list(self.thread_executor.map(func, locs))"""

        if False and self.particle_scale_factor != 1.0:
            scaled_locs = []
            for loc in locs:
                width_difference = int(loc.width * self.particle_scale_factor)
                height_difference = int(loc.height * self.particle_scale_factor)

                new_width = loc.width + width_difference
                new_height = loc.height + height_difference
                new_x = loc.x - (width_difference / 2)
                new_y = loc.y - (height_difference / 2)
                scaled_locs.append(Rect(new_x, new_y, new_width, new_height))

                new_width = loc.width - width_difference
                new_height = loc.height - height_difference
                new_x = loc.x + (width_difference / 2)
                new_y = loc.y + (height_difference / 2)
                scaled_locs.append(Rect(new_x, new_y, new_width, new_height))

                width_difference = self.initial_location.width - loc.width
                height_difference = self.initial_location.height - loc.height
                new_width = loc.width + width_difference
                new_height = loc.height + height_difference
                new_x = loc.x - (width_difference / 2)
                new_y = loc.y - (height_difference / 2)
                scaled_locs.append(Rect(new_x, new_y, new_width, new_height))

            locs.extend(scaled_locs)

        # if scaling is enabled, punish pixels with low feature rating
        punish_low = self.particle_scale_factor != 1.0
        slices = [img_mask[round(pos.top / scale_factor[1]):round((pos.bottom - 1) / scale_factor[1]),
                  round(pos.left / scale_factor[0]):round((pos.right - 1) / scale_factor[0])] for pos in locs]

        ps.append(time.time())  # 6

        sums = list(self.thread_executor.map(np.sum, slices))
        ps.append(time.time())  # 7

        quals = [self.position_quality(pos, frame.roi, img_mask_sum, inner_sum, scale_factor) / total_max
                 for pos, inner_sum in zip(locs, sums)]

        ps.append(time.time())  # 8

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

        ps.append(time.time())  # 9

        # i = i+1 - i

        ts = [p1 - p0 for p0,p1 in zip(ps[:-1], ps[1:])]

        total = ps[-1] - ps[0]
        log = ""
        for n, t in enumerate(ts):
            log += "; part{}: {:.2} ({:.2})".format(n, t, t / total)
        print(log[2:])

        return frame.predicted_position


    @staticmethod
    def calculate_sum(mat, punish_low=False):
        if punish_low:
            return np.multiply(mat - 0.2, 3, where=(mat < 0)).sum()
        else:
            return mat.sum()

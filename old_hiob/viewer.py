"""
Created on 2016-11-01

@author: Peer Springstübe
"""
import logging
import tkinter as tk
from PIL import ImageTk, ImageDraw, ImageFont

from hiob import data_set


logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

font_size = 16
font = ImageFont.truetype(
    "/usr/share/fonts/opentype/freefont/FreeMonoBold.otf", font_size)


def prepare_frame(ds, n, maxn):
    image = ds.images[n].copy()
    gt = ds.ground_truth[n]
    draw = ImageDraw.Draw(image)
    # draw frame number:
    draw.text((0, 0), "%s/%s" %
              (ds.set_name, ds.sample_name), "white", font=font)
    # draw frame number:
    draw.text((0, font_size), "#%04d/%04d" %
              (n, maxn), "white", font=font)
    if gt:
        draw.rectangle(gt.outer, None, (255, 255, 255, 255))
    return image


def prepare_frames(ds):
    frames = []
    for n in range(len(ds.images)):
        frames.append(prepare_frame(ds, n, len(ds.images)))
    return frames

ds = data_set.load_tb100_sample("Jogging.1")
frames = prepare_frames(ds)
frames[5].show()
# ds.images[0].show()

"""
Created on 2016-11-30

@author: Peer Springstübe
"""

import numpy as np
from PIL import Image


def figure_to_data(fig):
    """
    http://www.icare.univ-lille1.fr/node/1141
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def figure_to_image(fig):
    """
    http://www.icare.univ-lille1.fr/node/1141
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = figure_to_data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

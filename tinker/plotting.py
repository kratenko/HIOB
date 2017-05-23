import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fig2data(fig):
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


def fig2img(fig):
    """
    http://www.icare.univ-lille1.fr/node/1141
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


a = [0, 10, 4, 23, 15, 11, 12, 13, 15, 5, 13, 20, 5, 28, 24, 11, 6]

dim = np.arange(1, len(a) + 1)
f = plt.figure()
plt.axhline(y=20, color='r', linestyle='--')
plt.plot(dim, a, 'k', dim, a, 'bo')
plt.xlim(1, len(a))
im = fig2img(f)
im.show()
# plt.show()
exit()

plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3')  # subplot 211 title

plt.show()

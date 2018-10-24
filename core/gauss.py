import numpy as np
import scipy.stats as st

# https://gist.github.com/andrewgiessel/4635563


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def g1(x):
    return gaussian(x, 1, 1)


def g2(x, y):
    my = 0
    sig = 1
    return gaussian(x, my, sig) * gaussian(y, my, sig)


def gen_2d_gauss(xmy, xsigma, ymy, ysigma):
    return lambda x, y:  gaussian(x, xmy, xsigma) * gaussian(y, ymy, ysigma)


def gauss_dist(w, h, sigf=0.5):
    my_gauss = gen_2d_gauss((w - 1) * 0.5, w * sigf, (h - 1) * 0.5, h * sigf)
    return np.fromfunction(my_gauss, (w, h))


def xxgen_gauss_mask(mask_size, gauss_pos, sigf=0.5):
    mask = np.zeros(mask_size)
    m_pos = gauss_dist(*gauss_pos[2:4], sigf=sigf)
    x1, y1 = gauss_pos[0:2]
    x2 = x1 + gauss_pos[2]
    y2 = y1 + gauss_pos[3]
    print(mask.shape, x1, x2, y1, y2)
    mask[x1:x2, y1:y2] = m_pos
    return mask


def gen_gauss_mask(mask_size, gauss_pos, sigf=0.5):
    #print("GAUSS:", mask_size, gauss_pos)
    mask = np.zeros(mask_size)
    m_pos = gauss_dist(*gauss_pos[2:4], sigf=sigf)
    x1, y1 = gauss_pos[0:2]
    x2 = x1 + gauss_pos[2]
    y2 = y1 + gauss_pos[3]
    #print(mask.shape, x1, y1, x2, y2)
    mask[x1:x2, y1:y2] = m_pos
    return mask

"""
exit()

size = (64, 34)
#a = gauss_dist(*size, sigf=0.8)
a = gen_gauss_mask((480, 270), (269, 75, 4, 7), 0.5)
roi_size = (368, 368)
feature_size = (46, 46)

a = gen_gauss_mask((480, 270), (269, 75, 34, 64), 0.5)
#a = gen_gauss_mask((46, 46), (10, 10, 4, 7), 0.5)
print(a)

import matplotlib.pyplot as plt
plt.imshow(a.T, interpolation='none')
plt.show()
exit()

exit()

aa = gkern(101, nsig=1)
print(aa)
aa = imresize(aa, (64, 34))
print(aa)

import matplotlib.pyplot as plt
plt.imshow(aa, interpolation='none')
plt.show()
exit()

i_size = (480, 270)
pos = (269, 75, 34, 64)
p_size = pos[2:4]

m = np.zeros(i_size)
scale = min(p_size) / 3.0
"""

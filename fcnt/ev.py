from vis.DataSample import DataSample
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = '/data/Peer/data/'


def center(r):
    x, y, w, h = r
    return (x + w / 2, y + h / 2)


def rect_pixels(r):
    _, _, w, h = r
    return w * h


def intersect_rect(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x = max(x1, x2)
    y = max(y1, y2)
    w = max(min(x1 + w1, x2 + w2) - x, 0)
    h = max(min(y1 + h1, y2 + h2) - y, 0)
    return (x, y, w, h)


def overlap(r1, r2):
    ri = intersect_rect(r1, r2)
    print(ri)
    inter_size = rect_pixels(ri)
    if inter_size == 0:
        return 0.0
    union_size = rect_pixels(r1) + rect_pixels(r2) - inter_size
    return inter_size / union_size


def distance(r1, r2):
    x1, y1 = center(r1)
    x2, y2 = center(r2)
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def load_dir_sample(name):
    dir_path = os.path.join(DATA_PATH, 'tb100_unzipped', name)
    print("loading", dir_path)
    ds = DataSample()
    ds.load_from_dir(dir_path)
    # print ds.groundtruth_paths
    # print ds.groundtruths
    return ds


def build_dist_fun(dists):
    def f(thresh):
        smalls = [z for z in dists if z <= thresh]
        return len(smalls) / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        bigs = [z for z in overs if z >= thresh]
        return len(bigs) / len(overs)
    return f


s = load_dir_sample("Basketball")

gt = s.groundtruths[0]
fc = s.fcnt_position

dists = []
overs = []

for i in range(len(gt)):
    g = gt[i]
    f = fc[i]
    dist = distance(g, f)
    over = overlap(g, f)
    dists.append(dist)
    overs.append(over)
    print(g, rect_pixels(g), rect_pixels(f), dist, over)

dfun = build_dist_fun(dists)
ofun = build_over_fun(overs)


plt.figure(1)
plt.subplot(121)
x = np.arange(0., 50., 1.)
y = [dfun(a) for a in x]
plt.plot(x, y)

plt.subplot(122)
x = np.arange(0., 1., 0.01)
y = [ofun(a) for a in x]
auc = np.trapz(y, x)
tx = "AUC = %0.4f" % auc
plt.text(0.05, 0.05, tx)
print("AUC", auc, tx)
plt.plot(x, y)

plt.show()

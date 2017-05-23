from vis.DataSample import DataSample
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.io

from hiob.rect import Rect

DATA_PATH = '/data/Peer/data/'


def center(r):
    """
    Return center of given rect.
    """
    x, y, w, h = r
    return (x + w / 2, y + h / 2)


def rect_pixels(r):
    """
    Return number of pixels within rect (area).
    """
    _, _, w, h = r
    return w * h


def intersect_rect(r1, r2):
    """
    Return intersect rect for two rects.
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x = max(x1, x2)
    y = max(y1, y2)
    w = max(min(x1 + w1, x2 + w2) - x, 0)
    h = max(min(y1 + h1, y2 + h2) - y, 0)
    return (x, y, w, h)


def overlap(r1, r2):
    """
    Return relation overlapping part for two rects.
    """
    ri = intersect_rect(r1, r2)
    print(ri)
    inter_size = rect_pixels(ri)
    if inter_size == 0:
        return 0.0
    union_size = rect_pixels(r1) + rect_pixels(r2) - inter_size
    return inter_size / union_size


def distance(r1, r2):
    """
    Return distance of centers for two rects.
    """
    x1, y1 = center(r1)
    x2, y2 = center(r2)
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def load_csv(path):
    p = re.compile("(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
    csv = []
    with open(path) as f:
        for line in f:
            m = p.match(line)
            r = [int(x) for x in m.groups()]
            # print(",".join(m.groups()))
            csv.append(r)
    return csv


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


def load_mat(path):
    mpos = scipy.io.loadmat(path)['position']
    rs = []
    for i in range(len(mpos[0])):
        geo = mpos[0][i], mpos[1][i], mpos[2][i], mpos[4][i]
        loc = affgeo2loc(geo)
        loc = [round(x) for x in loc]
        rs.append(loc)
        # print(geo, loc)
        # print(loc)
    return rs


def load_sample(name):
    print(name)
    csv = load_csv(
        os.path.join(DATA_PATH, 'tb100_unzipped', name, 'groundtruth_rect.txt'))
    mat = load_mat(
        os.path.join(DATA_PATH, 'tb100_unzipped', name, 'position.mat'))
    return csv, mat

# load_csv('/data/Peer/data/tb100_unzipped/Jump/groundtruth_rect.txt')
# load_mat('/data/Peer/data/tb100_unzipped/MotorRolling/position.mat')
# exit()


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
        #smalls = [z for z in dists if z <= thresh]
        return (dists <= thresh).sum() / len(dists)
        #smalls = [z for z in dists if z <= thresh]
        # return len(smalls) / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
        bigs = [z for z in overs if z >= thresh]
        return len(bigs) / len(overs)
    return f


#s = load_dir_sample("MotorRolling")
#gt = s.groundtruths[0]
#fc = s.fcnt_position

gt = []
fc = []
samples = ['Basketball', 'Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1',
           'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl',
           'Board', 'Bolt', 'Bolt2', 'Box', 'Boy', 'Car1', 'Car2', 'Car4',
           'Car24', 'CarDark', 'CarScale', 'ClifBar', 'Coke', 'Couple',
           'Coupon', 'Crossing', 'Crowds', 'Dancer', 'Dancer2',  # 'David',
           'David2', 'David3', 'Deer', 'Diving', 'Dog', 'Dog1', 'Doll',
           'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish',
           'FleetFace', 'Football', 'Football1',
           'Freeman1', 'Freeman3', 'Freeman4',
           'Girl', 'Girl2', 'Gym',
           'Human2', 'Human3', 'Human4',  # Human4
           'Human5', 'Human6', 'Human7', 'Human8', 'Human9',
           'Ironman',  # Jogging
           'Jump', 'Jumping', 'KiteSurf', 'Lemming',
           'Liquor', 'Man', 'Matrix', 'Mhyang',
           'MotorRolling', 'MountainBike',
           'Panda', 'RedTeam', 'Rubik', 'Shaking',
           'Singer1', 'Singer2', 'Skater', 'Skater2',
           'Skating1',  # Skating2
           'Skiing', 'Soccer', 'Subway', 'Surfer',
           'Suv', 'Sylvester', 'Tiger1', 'Tiger2',
           'Toy', 'Trans', 'Trellis', 'Twinnings',
           'Vase', 'Walking', 'Walking2', 'Woman',
           ]
for name in samples:
    gt_, fc_ = load_sample(name)
    gt.extend(gt_)
    fc.extend(fc_)

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

dists = np.array(dists)
overs = np.array(overs)

dfun = build_dist_fun(dists)
ofun = build_over_fun(overs)


plt.figure(1)
plt.subplot(121)
x = np.arange(0., 50.1, .1)
y = [dfun(a) for a in x]
at20 = dfun(20)
tx = "prec(20) = %0.4f" % at20
plt.text(5.05, 0.05, tx)
#print("AUC", auc, tx)
plt.plot(x, y)

plt.subplot(122)
x = np.arange(0., 1.001, 0.001)
y = [ofun(a) for a in x]
auc = np.trapz(y, x)
tx = "AUC = %0.4f" % auc
plt.text(0.05, 0.05, tx)
print("AUC", auc, tx)
plt.plot(x, y)

plt.show()

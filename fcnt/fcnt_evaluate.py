"""
Created on 2016-12-14

@author: Peer Springstübe
"""
from hiob.rect import Rect
import matplotlib

RUN_DIR = "/data/Peer/FCNT-eval/run3"
GT_DIR = "/data/Peer/data/tb100_unzipped"
ENDING = '.position.mat'

import os
import scipy.io
import numpy as np
import re
from matplotlib import pyplot as plt, matplotlib_fname

tb50 = """Ironman
Matrix
MotorRolling
Soccer
Skiing
Freeman4
Freeman1
Skating1
Tiger2
Liquor
Coke
Football
FleetFace
Couple
Tiger1
Woman
Bolt
Freeman3
Basketball
Lemming
Singer2
Subway
CarScale
David3
Shaking
Sylvester
Girl
Jumping
Trellis
David
Boy
Deer
FaceOcc2
Dudek
Football1
Suv
Jogging.1
Jogging.2
MountainBike
Crossing
Singer1
Dog1
Walking
Walking2
Doll
Car4
David2
CarDark
Mhyang
FaceOcc1
Fish"""

tb50_list = tb50.split("\n")
tb50_set = set(tb50_list)


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


def load_rects(path):
    a = scipy.io.loadmat(path)
    mrects = a['rects']
    r = []
    for i in range(len(mrects[0])):
        mr = mrects[0][i], mrects[1][i], mrects[2][i], mrects[3][i]
        r.append(Rect(mr))
    return r


def load_gt(sname):
    path = os.path.join(GT_DIR, sname, 'groundtruth_rect.txt')
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

all_rs = []
all_gts = []
tb50_rs = []
tb50_gts = []
tbnew_rs = []
tbnew_gts = []

run_dirs = ["/data/Peer/FCNT-eval/run1",
            "/data/Peer/FCNT-eval/run2", "/data/Peer/FCNT-eval/run3"]
#run_dirs = ["/data/Peer/FCNT-eval/run3", ]
for run_dir in run_dirs:
    for fname in os.listdir(run_dir):
        if not fname.endswith(ENDING):
            continue
        sname = fname[:-len(ENDING)]

        rs_path = os.path.join(run_dir, sname + ENDING)
        rs = load_rects(rs_path)
        gts = load_gt(sname)
        if sname == 'Football1':
            rs = rs[0:74]
        elif sname == 'Freeman4':
            rs = rs[0:283]
        elif sname == 'Freeman3':
            rs = rs[0:460]
        elif sname == 'Diving':
            rs = rs[0:215]
        elif sname == 'David':
            rs = rs[299:]
            gts = gts[299:]
        assert len(rs) == len(gts)
        all_rs.extend(rs)
        all_gts.extend(gts)
        if sname in tb50_set:
            tb50_rs.extend(rs)
            tb50_gts.extend(gts)
        else:
            tbnew_rs.extend(rs)
            tbnew_gts.extend(gts)
        print("Sample:", sname, len(rs), len(gts), len(all_gts))
print(len(all_rs), len(tb50_rs))
# print(all_rs)
# print(all_gts)
cd = np.empty(len(all_rs))
ov = np.empty(len(all_gts))
print(len(cd), len(ov))
for n, (r, gt) in enumerate(zip(all_rs, all_gts)):
    if gt is None:
        print(n)
    cd[n] = r.center_distance(gt)
    ov[n] = r.overlap_score(gt)
dfun = build_dist_fun(cd)
ofun = build_over_fun(ov)

tb50_cd = np.empty(len(tb50_rs))
tb50_ov = np.empty(len(tb50_gts))
print(len(tb50_cd), len(tb50_ov))
for n, (r, gt) in enumerate(zip(tb50_rs, tb50_gts)):
    if gt is None:
        print(n)
    tb50_cd[n] = r.center_distance(gt)
    tb50_ov[n] = r.overlap_score(gt)
tb50_dfun = build_dist_fun(tb50_cd)
tb50_ofun = build_over_fun(tb50_ov)

tbnew_cd = np.empty(len(tbnew_rs))
tbnew_ov = np.empty(len(tbnew_gts))
print(len(tbnew_cd), len(tbnew_ov))
for n, (r, gt) in enumerate(zip(tbnew_rs, tbnew_gts)):
    if gt is None:
        print(n)
    tbnew_cd[n] = r.center_distance(gt)
    tbnew_ov[n] = r.overlap_score(gt)
tbnew_dfun = build_dist_fun(tbnew_cd)
tbnew_ofun = build_over_fun(tbnew_ov)

dim = np.arange(1, len(cd) + 1)

figsize = (6, 5)

f = plt.figure(figsize=figsize)
plt.title("FCNT - 3 evaluations, precision")
plt.xlabel("center distance [pixels]")
plt.ylabel("occurrence")
plt.xlim(xmin=0, xmax=50)
plt.ylim(ymin=0.0, ymax=1.0)
plt.axvline(x=20, linestyle=':', color='black')
x = np.arange(0., 50.1, .1)
y = [dfun(a) for a in x]
at20 = dfun(20)
#tx = "prec(20) = %0.4f" % at20
#plt.text(5.05, 0.05, tx)
plt.plot(x, y, 'b-')

tb50_y = [tb50_dfun(a) for a in x]
tb50_at20 = tb50_dfun(20)
#tx = "prec(20) = %0.4f" % tb50_at20
#plt.text(5.05, 0.10, tx)
plt.plot(x, tb50_y, 'r-')

tbnew_y = [tbnew_dfun(a) for a in x]
tbnew_at20 = tbnew_dfun(20)
plt.plot(x, tbnew_y, 'orange')

# tb50_h = matplotlib.patches.Patch(
#    color='red', label='tb100(p50) - prec(20) = %0.3f' % tb50_at20)
# all_h = matplotlib.patches.Patch(
#    color='blue', label='tb100(full)  - prec(20) = %0.3f' % at20)
# tbnew_h = matplotlib.patches.Patch(
#    color='orange', label='tb100(n50) - prec(20) = %0.3f' % tbnew_at20)
tb50_h = matplotlib.patches.Patch(
    color='red', label='%0.3f - tb100(p50)' % tb50_at20)
all_h = matplotlib.patches.Patch(
    color='blue', label='%0.3f - tb100(full)' % at20)
tbnew_h = matplotlib.patches.Patch(
    color='orange', label='%0.3f - tb100(n50)' % tbnew_at20)
plt.legend(handles=[tb50_h, all_h, tbnew_h], loc='lower right')

plt.savefig("precision_plot.pdf")
plt.savefig("precision_plot.svg")
plt.show()


f = plt.figure(figsize=figsize)
x = np.arange(0., 1.001, 0.001)
y = [ofun(a) for a in x]
tb50_y = [tb50_ofun(a) for a in x]
tbnew_y = [tbnew_ofun(a) for a in x]
auc = np.trapz(y, x)
tb50_auc = np.trapz(tb50_y, x)
tbnew_auc = np.trapz(tbnew_y, x)
#tx = "AUC = %0.4f" % auc
#plt.text(0.05, 0.05, tx)
plt.title("FCNT - 3 evaluations, success")
plt.xlabel("overlap score")
plt.ylabel("occurrence")
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y, 'b')
plt.plot(x, tb50_y, 'r')
plt.plot(x, tbnew_y, 'orange')
tb50_h = matplotlib.patches.Patch(
    color='red', label='%0.3f - tb100(p50)' % tb50_auc)
all_h = matplotlib.patches.Patch(
    color='blue', label='%0.3f - tb100(full)' % auc)
new_h = matplotlib.patches.Patch(
    color='orange', label='%0.3f - tb100(n50)' % tbnew_auc)
plt.legend(handles=[tb50_h, all_h, new_h], loc='lower left')
plt.savefig("success_plot.pdf")
plt.savefig("success_plot.svg")
plt.show()

# tb100/p50


exit()

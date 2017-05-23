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
import yaml

#fname = "tracking_log_one_coke.txt"
fname = "/data/Peer/thetrack1.txt"

# center dists
cds = []
# overlap scores
ovs = []

# load from tracking log csv
with open(fname) as f:
    for line in f.readlines():
        parts = line.strip().split(",")
        n = int(parts[0])
        qual = float(parts[5])
        dist = float(parts[10])
        over = float(parts[11])
        cds.append(dist)
        ovs.append(over)

cd = np.asarray(cds)
ov = np.asarray(ovs)


print(len(cd), len(ov))


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f

dfun = build_dist_fun(cd)
ofun = build_over_fun(ov)


dim = np.arange(1, len(cd) + 1)

figsize = (6, 5)

f = plt.figure(figsize=figsize)
plt.title("HIOB, no resize, no update, precision plot")
plt.xlabel("center distance [pixels]")
plt.ylabel("occurrence")
plt.xlim(xmin=0, xmax=50)
plt.ylim(ymin=0.0, ymax=1.0)
x = np.arange(0., 50.1, .1)
y = [dfun(a) for a in x]
at20 = dfun(20)
#tx = "prec(20) = %0.4f" % at20
#plt.text(5.05, 0.05, tx)
plt.plot(x, y, 'b-')


all_h = matplotlib.patches.Patch(
    color='blue', label='tb100(p50)  - prec(20) = %0.3f' % at20)
plt.legend(handles=[all_h, ], loc='lower right')

plt.savefig("precision_plot.pdf")
plt.savefig("precision_plot.svg")
plt.show()


f = plt.figure(figsize=figsize)
x = np.arange(0., 1.001, 0.001)
y = [ofun(a) for a in x]
auc = np.trapz(y, x)

#tx = "AUC = %0.4f" % auc
#plt.text(0.05, 0.05, tx)
plt.title("HIOB, no resize, no update, success plot")
plt.xlabel("overlap score")
plt.ylabel("occurrence")
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y, 'b')
all_h = matplotlib.patches.Patch(
    color='blue', label='tb100(p50)  - AUC = %0.3f' % auc)
plt.legend(handles=[all_h, ], loc='lower left')
plt.savefig("success_plot.pdf")
plt.savefig("success_plot.svg")
plt.show()

# tb100/p50


exit()

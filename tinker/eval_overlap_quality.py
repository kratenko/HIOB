"""
Created on 2017-03-01

@author: Peer Springstübe
"""

apath = "/data/Peer/hiob_runs/goodrun,noupdate2,sigma08/hiob-execution-wtmgws2-2017-02-23-17.38.09.653900"

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

OVER_LOST_LIMIT = 0.0

#fname = "tracking_log_one_coke.txt"
fname = "/data/Peer/thetrack1.txt"
vals = []
with open(fname) as f:
    for line in f.readlines():
        parts = line.strip().split(",")
        n = int(parts[0])
        qual = float(parts[5])
        over = float(parts[11])
        lost = over < OVER_LOST_LIMIT
        #print("%d: %f, %s" % (n, qual, lost))
        vals.append([qual, over])

qualover = np.asarray(vals)


def build_over_fun1(overs, limit):
    def f(thresh):
        hits = 0
        fails = 0
        for v in overs:
            if v[0] >= thresh:
                if v[1] > limit:
                    hits += 1
                else:
                    fails += 1
        #print(hits, fails)
        if hits + fails == 0:
            return 0.0
        return hits / (hits + fails)
    return f


def build_over_fun2(overs, limit):
    def f(thresh):
        hits = 0
        fails = 0
        for v in overs:
            if v[0] < thresh:
                if v[1] > limit:
                    hits += 1
                else:
                    fails += 1
        #print(hits, fails)
        if hits + fails == 0:
            return 0.0
        return hits / (hits + fails)
    return f

ofun1 = build_over_fun1(qualover, OVER_LOST_LIMIT)
ofun2 = build_over_fun2(qualover, OVER_LOST_LIMIT)

figsize = (6, 5)

f = plt.figure(figsize=figsize)

plt.title("Quality of confidence rating")
plt.xlabel("confidence threshold")
plt.ylabel("predictions overlapping")

tb50_h = matplotlib.patches.Patch(
    color='red', label='confidence ≥ threshold')
all_h = matplotlib.patches.Patch(
    color='blue', label='confidence < threshold')
plt.legend(handles=[tb50_h, all_h, ], loc='lower right')

# plt.subplot(122)
x = np.arange(0., 1.001, 0.001)
y = [ofun1(a) for a in x]
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y, 'r-')

y = [ofun2(a) for a in x]
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y, 'b-')

plt.savefig("confidence_quality.pdf")
plt.savefig("confidence_quality.svg")

plt.show()

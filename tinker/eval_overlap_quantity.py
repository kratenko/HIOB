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
quals = qualover[:, 0]
rn = quals.round(2)


def build_fun(rns):
    def f(val):
        return np.sum(rn == val)
    return f

fun = build_fun(rn)

figsize = (6, 5)

f = plt.figure(figsize=figsize)

plt.title("Quantity of confidence rating")
plt.xlabel("confidence (rounded)")
plt.ylabel("occurrences")

t1 = matplotlib.patches.Patch(
    color=None,
    label='confidence ≥ threshold')
#plt.legend(handles=[t1, ], loc='upper right')

plt.annotate('total values: %d' % len(rn), xy=(.5, 1800))

# plt.subplot(122)
x = np.arange(0., 1.01, 0.01)
y = [fun(a) for a in x]
#plt.xlim(xmin=0.0, xmax=1.0)
#plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y, 'ro')


plt.savefig("confidence_quantity.pdf")
plt.savefig("confidence_quantity.svg")

plt.show()

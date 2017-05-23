"""
Created on 2016-10-04

@author: Peer Springstübe
"""

import os

files = os.listdir(".")
files.sort()

for n, fname in enumerate(files):
    tname = "%04d.jpg" % (n + 1)
    print("{}->{}".format(fname, tname))
    os.rename(fname, tname)

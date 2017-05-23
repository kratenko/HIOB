"""
Created on 2016-12-14

@author: Peer Springstübe
"""

import datetime
import re

durs = []
total = datetime.timedelta()
r = re.compile(r".*'(\d+:\d\d:\d\d\.\d+)'.*")

with open("/data/Peer/FCNT-eval/run1/durations50.log") as f:
    for line in f.readlines():
        line = line.strip()
        line = line[10:]
        m = r.match(line)
        if m:
            s = m.group(1)
            t = datetime.datetime.strptime(s, "%H:%M:%S.%f")
            d = datetime.timedelta(
                hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
            print(s, t, d)
            durs.append(d)
            total = total + d
print(total, total.total_seconds())

"""
Created on 2016-08-25

@author: Peer Springstübe
"""
from datetime import timedelta


def dur(m, s):
    return timedelta(minutes=int(m), seconds=int(s))

s = """2,29
3,29
2,37
1,51
3,23
3,46
4,05
3,16
3,32
3,39
1,03
3,27
3,09
3,44
4,42
4,29"""

ss = s.split("\n")

durs = [dur(*(xx.split(","))) for xx in ss]

print(durs)
toto = sum([_.seconds for _ in durs])
print(toto % 60, (toto - toto % 60) // 60)

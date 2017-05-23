"""
Created on 2016-11-22

@author: Peer Springstübe
"""
from collections import OrderedDict

o = OrderedDict()
o['a'] = 1
o['b'] = 2
o['c'] = 3
o['d'] = 4
o['e'] = 5
o['f'] = 6
o['g'] = 7
o['h'] = 8
o['i'] = 9
u = {k: v for k, v in o.items()}
print(list(o.keys()))
print(list(o.values()))

print(list(u.keys()))
print(list(u.values()))

exit()

from hiob.vgg16 import Vgg16

v = Vgg16(
    input_size=(368, 368),
    vgg16_npy_path='/informatik2/students/home/3springs/git/tensorflow-vgg/vgg16.npy')

for k, v in v.features.items():
    print(k, v, v.get_shape().as_list())

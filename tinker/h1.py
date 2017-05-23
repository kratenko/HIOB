"""
Created on 2016-12-15

@author: Peer Springstübe
"""

import hyperopt
import hyperopt.pyll.stochastic
from hyperopt import hp

space = [
    hp.uniform('x', -10, 10),
    hp.quniform('y', 10, 100, 1),
]

for _ in range(10):
    print(hyperopt.pyll.stochastic.sample(space))

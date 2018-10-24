"""
Created on 2016-11-29

@author: Peer Springstübe
"""
from .. import HiobModule


class Pursuer(HiobModule.HiobModule):

    def pursue(self, state, frame):
        raise NotImplementedError()
"""
Created on 2016-10-06

@author: Peer Springstübe
"""

import numpy as np


class Rect(object):
    """
    Hopefully helpful rectangle class, heavily inspired by pygame.
    """
    __slots__ = ['_x', '_y', '_w', '_h']
    #_x = None
    #_y = None
    #_w = None
    #_h = None

    def __init__(self, *args):
        if len(args) == 4:
            self._x = int(args[0])
            self._y = int(args[1])
            self._w = int(args[2])
            self._h = int(args[3])
        elif len(args) == 2:
            if type(args[0]) == type(args[1]) == int:
                self._x = 0
                self._y = 0
                self._w = int(args[0])
                self._h = int(args[1])
            else:
                self._x, self._y = int(args[0][0]), int(args[0][1])
                self._w, self._h = int(args[1][0]), int(args[1][1])
        elif len(args) == 1:
            if isinstance(args[0], Rect):
                self._x, self._y, self._w, self._h = args[0]
            else:
                # list or tuple
                if len(args[0]) == 4:
                    self._x, self._y, self._w, self._h = tuple(
                        int(a) for a in args[0])
                elif len(args[0]) == 2:
                    self._x, self._y = 0, 0
                    self._w, self._h = int(args[0][0]), int(args[0][1])
        if self._w is None:
            raise TypeError("Must be rect like")

    def __getattr__(self, name):
        if name == 'top' or name == "y":
            return self._y
        elif name == 'left' or name == "x":
            return self._x
        elif name == 'bottom':
            return self._y + self._h
        elif name == 'right':
            return self._x + self._w
        elif name == 'topleft':
            return self._x, self._y
        elif name == 'bottomleft':
            return self._x, self._y + self._h
        elif name == 'topright':
            return self._x + self._w, self._y
        elif name == 'bottomright':
            return self._x + self._w, self._y + self._h
        elif name == 'midtop':
            return self._x + self._w // 2, self._y
        elif name == 'midleft':
            return self._x, self._y + self._h // 2
        elif name == 'midbottom':
            return self._x + self._w // 2, self._y + self._h
        elif name == 'midright':
            return self._x + self._w, self._y + self._h // 2
        elif name == 'center':
            return self._x + self._w // 2, self._y + self._h // 2
        elif name == 'centerx':
            return self._x + self._w // 2
        elif name == 'centery':
            return self._y + self._h // 2
        elif name == 'size':
            return self._w, self._h
        elif name == 'width' or name == "w":
            return self._w
        elif name == 'height' or name == "h":
            return self._h
        elif name == 'tuple':
            return self._x, self._y, self._w, self._h
        elif name == 'outer':
            return self._x, self._y, self._x + self._w, self._y + self._h
        elif name == 'inner':
            return self._x, self._y, self._x + self._w - 1, self._y + self._h - 1
        else:
            raise AttributeError

    def __len__(self):
        return 4

    def __setattr__(self, name, value):
        value = int(round(value))
        if name[0] == '_':
            return object.__setattr__(self, name, value)

        if name == 'top' or name == 'y':
            self._y = value
        elif name == 'left' or name == 'x':
            self._x = value
        elif name == 'bottom':
            self._y = value - self._h
        elif name == 'right':
            self._x = value - self._w
        elif name == 'topleft':
            self._x, self._y = value
        elif name == 'bottomleft':
            self._x = value[0]
            self._y = value[1] - self._h
        elif name == 'topright':
            self._x = value[0] - self._w
            self._y = value[1]
        elif name == 'bottomright':
            self._x = value[0] - self._w
            self._y = value[1] - self._h
        elif name == 'midtop':
            self._x = value[0] - self._w / 2
            self._y = value[1]
        elif name == 'midleft':
            self._x = value[0]
            self._y = value[1] - self._h / 2
        elif name == 'midbottom':
            self._x = value[0] - self._w / 2
            self._y = value[1] - self._h
        elif name == 'midright':
            self._x = value[0] - self._w
            self._y = value[1] - self._h / 2
        elif name == 'center':
            self._x = value[0] - self._w / 2
            self._y = value[1] - self._h / 2
        elif name == 'centerx':
            self._x = value - self._w / 2
        elif name == 'centery':
            self._y = value - self._h / 2
        elif name == 'size':
            self._w, self._h = value
        elif name == 'width' or name == "w":
            self._w = value
        elif name == 'height' or name == "h":
            self._h = value
        else:
            raise AttributeError

    def __getitem__(self, key):
        return (self._x, self._y, self._w, self._h)[key]

    def __setitem__(self, key, value):
        r = [self._x, self._y, self._w, self._h]
        r[key] = value
        self._x, self._y, self._w, self._h = r

    def __repr__(self):
        return '<rect(%d, %d, %d, %d)>' % \
            (self._x, self._y, self._w, self._h)

    def copy(self):
        return Rect(self)

    # metrics...
    def center_distance(self, other):
        """
        Return eucledian distance from center to other rect's center.  
        """
        self_center = np.array(self.center)
        other_center = np.array(other.center)
        return np.linalg.norm(self_center - other_center)

    def pixel_count(self):
        """
        Return number of pixels inside rect.
        """
        return self.width * self.height

    def intersect(self, other):
        """
        Return new Rect of intersected area or None if exclusive.
        """
        x1, y1, w1, h1 = self
        x2, y2, w2, h2 = other
        x = max(x1, x2)
        y = max(y1, y2)
        w = max(min(x1 + w1, x2 + w2) - x, 0)
        h = max(min(y1 + h1, y2 + h2) - y, 0)
        if w == 0 or h == 0:
            return None
        return Rect(x, y, w, h)

    def overlap_score(self, other):
        """
        Return overlap score of this rect with other rect.

        See definition in chapter 4 of paper "Online Object Tracking: A Benchmark" by Wu et al., 2013.
        """
        intersect = self.intersect(other)
        if intersect is None:
            # no intersection, score is zero
            return 0.0
        inter_size = intersect.pixel_count()
        union_size = self.pixel_count() + other.pixel_count() - inter_size
        return inter_size / union_size

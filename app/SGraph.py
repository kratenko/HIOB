import collections

from PIL import Image
from PIL.ImageDraw import Draw


class SGraph(object):

    def __init__(self, min_y=0.0, max_y=1.0, length=10, height=20):
        self.store = collections.deque([None] * length)
        self.length = length
        self.min_y = min_y
        self.max_y = max_y
        self.height = height
        self.size = (self.length, height)
        self.image = None
        self.dirty = True
        self.ylines = []

    def append(self, value):
        self.store.append(value)
        while len(self.store) > self.length:
            self.store.popleft()
        self.dirty = True

    def create_image(self):
        im = Image.new("RGB", self.size, "white")
        draw = Draw(im)
        # add horizontal lines to show limits:
        for v in self.ylines:
            ry = 1 - (v - self.min_y) / (self.max_y - self.min_y)
            ry = ry * self.size[1]
            draw.line(((0, ry), (self.size[0], ry)), "green", 1)
        # draw values as connected dotes to create a graph
        last_pos = None
        for n, v in enumerate(self.store):
            if v is None:
                last_pos = None
                continue
            ry = 1 - (v - self.min_y) / (self.max_y - self.min_y)
            pos = (n, ry * self.size[1])
            if last_pos is None:
                draw.point(pos, "black")
            else:
                draw.line([last_pos, pos], "black", 1)
            last_pos = pos
        self.image = im
        self.dirty = False

    def get_image(self):
        if self.dirty:
            self.create_image()
        return self.image
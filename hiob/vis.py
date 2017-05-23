import logging
import transitions
from PIL import ImageTk, ImageDraw
import queue
import threading
from hiob.data_set import DataDirectory
from hiob.rect import Rect

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

#consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

logger = logging.getLogger(__name__)

from hiob.configuration import Configurator
conf = Configurator()

import tkinter as tk


class ImageLabel(tk.Label):

    def __init__(self, *args, **kwargs):
        tk.Label.__init__(self, *args, **kwargs)
        self._image = None

    def set_image(self, image):
        if image is None:
            self._image = None
            self['image'] = None
        else:
            self._image = ImageTk.PhotoImage(image)
            self['image'] = self._image


class AppTerminated(Exception):
    pass


class App:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hiob DataSet visualiser")

        self.frames = []
        self.current_frame = 0

        self.build_widgets()
        self.wait_time = int(1000 / 30)

        self.dd = DataDirectory('/data/Peer/data')
        sample = self.dd.get_sample('tb100', 'Tiger1')
#        sample = self.dd.get_sample('princeton', 'bag1')
        self.prepare_sample(sample)
        self.display_frame()
        self.root.after(self.wait_time, self.display_next_frame)

    def prepare_sample(self, sample):
        self.frames = []
        print("loading")
        sample.load()
        print(len(sample.images), len(sample.ground_truth))
        while len(sample.images) > len(sample.ground_truth):
            sample.ground_truth.append(None)
        for n, (im, gt) in enumerate(zip(sample.images, sample.ground_truth)):
            # make a copy, so sample stays untouched:
            im = im.copy()
            # draw GT and
            if gt:
                draw = ImageDraw.Draw(im)
                draw.rectangle(gt.outer, None, 'yellow')
                r2 = Rect(gt.left - 1, gt.top - 1, gt.width + 2, gt.height + 2)
                draw.rectangle(r2.outer, None, 'yellow')
            t = "Frame #%04d/%04d" % (n + 1, sample.actual_frames)
            self.frames.append([im, t])
            im.save("/tmp/vis/%04d.jpg" % n)

        self.sample_text['text'] = "Sample %s/%s, Attributes: %s" % (
            sample.set_name, sample.name, ', '.join(sample.attributes))

    def display_frame(self):
        if self.frames:
            (im, t) = self.frames[self.current_frame]
            self.video_image.set_image(im)
            self.video_text['text'] = t

    def display_next_frame(self):
        self.current_frame += 1
        if self.current_frame >= len(self.frames):
            self.current_frame = 0
        self.display_frame()
        self.root.after(self.wait_time, self.display_next_frame)

    def build_widgets(self):
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        self.sample_text = tk.Label(self.video_frame)
        self.sample_text.pack()
        self.video_text = tk.Label(self.video_frame)
        self.video_text.pack()
        self.video_image = ImageLabel(self.video_frame)
        self.video_image.pack()

    def run(self):
        self.root.mainloop()

app = App()
app.run()

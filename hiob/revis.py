import logging
import transitions
from PIL import ImageTk, ImageDraw, Image, ImageFont
import queue
import threading
from hiob.data_set import DataDirectory
import os
import pickle
from hiob.rect import Rect
import collections
from PIL.ImageDraw import Draw

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

#consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

logger = logging.getLogger(__name__)

from hiob.configuration import Configurator
conf = Configurator()

t_path = "/data/Peer/hiob_logs/hiob-execution-wtmpc27-2017-03-29-18.02.55.204522/tracking-0001-tb100-Deer"
#t_path = "/data/Peer/hiob_logs/hiob-execution-wtmpc27-2017-03-29-20.20.20.225711/tracking-0001-princeton-book_turn"


# t24 Board
# t18 Rubik
show_gt = True
base = "/data/Peer/hiob_total_recall"
tr = "t33"
sn = "Freeman4"

p1 = None
p2 = None
for d in os.listdir(base):
    if d.startswith(tr + '-'):
        p1 = os.path.join(base, d)
        break
if p1 is None:
    print("Could not find tracker log dir")
    exit()
for d in os.listdir(p1):
    if d.startswith('tracking-') and d.endswith('-' + sn):
        p2 = os.path.join(p1, d)
if p2 is None:
    print("Could not find tracking log dir")
    exit()
print("Found tracking at %s" % p2)
t_path = p2


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

# == coordinate conversions ==


class Conv(object):

    def __init__(self, mask_size=None):
        self.sroi_size = [368, 368]
        if mask_size is None:
            self.mask_size = [46, 46]
        else:
            self.mask_size = mask_size
        #
        self.sroi_to_mask_ratio = (
            self.sroi_size[0] / self.mask_size[0], self.sroi_size[1] / self.mask_size[1])

    def capture_to_sroi(self, pos, roi):
        """
        Convert rect in capture to rect in scaled roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        ix = round((px - rx) * scale_w)
        iy = round((py - ry) * scale_h)
        iw = scale_w * pw
        ih = scale_h * ph
        return Rect(ix, iy, iw, ih)

    def sroi_to_capture(self, pos, roi):
        """
        Convert rect in scaled roi to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        cx = round(sx / scale_w + rx)
        cy = round(sy / scale_h + ry)
        cw = sw / scale_w
        ch = sh / scale_h
        return Rect(cx, cy, cw, ch)

    def sroi_to_mask(self, sroi_pos):
        return Rect(
            int(sroi_pos.left / self.sroi_to_mask_ratio[0]),
            int(sroi_pos.top / self.sroi_to_mask_ratio[1]),
            int(sroi_pos.width / self.sroi_to_mask_ratio[0]),
            int(sroi_pos.height / self.sroi_to_mask_ratio[1]),
        )

    def mask_to_sroi(self, mask_pos):
        return Rect(
            int(mask_pos.left * self.sroi_to_mask_ratio[0]),
            int(mask_pos.top * self.sroi_to_mask_ratio[1]),
            int(mask_pos.width * self.sroi_to_mask_ratio[0]),
            int(mask_pos.height * self.sroi_to_mask_ratio[1]),
        )

    def capture_to_mask(self, pos, roi):
        """
        Convert rect in capture to rect in mask roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.sroi_to_mask_ratio[0]
        scale_h = (self.sroi_size[1] / rh) / self.sroi_to_mask_ratio[1]
        ix = int(round((px - rx) * scale_w))
        iy = int(round((py - ry) * scale_h))
        iw = int(round(scale_w * pw))
        ih = int(round(scale_h * ph))
        return Rect(ix, iy, iw, ih)

    def mask_to_capture(self, pos, roi):
        """
        Convert rect in mask to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.sroi_to_mask_ratio[0]
        scale_h = (self.sroi_size[1] / rh) / self.sroi_to_mask_ratio[1]
        cx = int(round(sx / scale_w + rx))
        cy = int(round(sy / scale_h + ry))
        cw = int(round(sw / scale_w))
        ch = int(round(sh / scale_h))
        return Rect(cx, cy, cw, ch)


class Run(object):

    def __init__(self, path):
        self.t_path = path
        self.p_path = os.path.join(self.t_path, 'tracking_log.p')
        self.e_path = os.path.join(self.t_path, 'evaluation.txt')

        ev = {}
        with open(self.e_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                a = line.split("=", 2)
                if len(a) == 2:
                    ev[a[0]] = a[1]
        self.evaluation = ev

        with open(self.p_path, "rb") as f:
            self.run = pickle.load(f)

import tkinter as tk

cc = Conv()


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
        self.root.title("Hiob Tracking revisualiser")

        self.frames = []
        self.current_frame = 0

        self.images = {}
        self.texts = {}
        self.build_widgets()
        self.fps = 30
        self.wait_time = int(1000 / self.fps)

        self.tracking_run = Run(t_path)

        self.dd = DataDirectory('/data/Peer/data')
#        sample = self.dd.get_sample('tb100', 'RedTeam')
        sample = self.dd.get_sample(
            self.tracking_run.evaluation['set_name'], self.tracking_run.evaluation['sample_name'])
        self.prepare_sample(sample)
        self.display_frame()
        self.root.after(self.wait_time, self.display_next_frame)

        self.running = True
        self.root.bind("<KeyPress>", self.keypress)

    def keypress(self, event):
        if event.keysym == 'space':
            # space - start/stop
            if self.running:
                self.running = False
            else:
                self.running = True
                self.display_next_frame()
        elif event.keysym == 'Right':
            if not self.running:
                self.display_next_frame()
        elif event.keysym == 'Left':
            if not self.running:
                self.display_previous_frame()
        elif event.keysym == 'Up':
            self.set_fps(self.fps + 1)
        elif event.keysym == 'Down':
            self.set_fps(self.fps - 1)
        elif event.keysym == 'Home':
            self.set_fps(30)
        elif event.keysym == 'Prior':
            self.set_fps(300)
        elif event.keysym == 'Next':
            self.set_fps(1)
        print(event.char, event.type, event.keycode, event.keysym)

    def set_fps(self, fps):
        self.fps = max(1, fps)
        print("Setting frame rate to %d fps" % self.fps)
        self.wait_time = int(1000 / self.fps)

    def prepare_sample(self, sample):
        self.frames = []
        self.heats = []
        self.srois = []
        l = sample.actual_frames
        self.confidence_plotter = SGraph(0.0, 1.0, l, 100)
        self.distance_plotter = SGraph(0, 100, l, 100)
        self.distance_plotter.ylines = [20]
        self.overlap_plotter = SGraph(0.0, 1.0, l, 100)
        self.update_plotter = Image.new("RGB", (l, 10), "white")
        udraw = ImageDraw.Draw(self.update_plotter)

        print("loading sample %s", sample.full_name)
        sample.load()
        print(len(sample.images), len(sample.ground_truth))
        while len(sample.images) > len(sample.ground_truth):
            sample.ground_truth.append(None)

        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/freefont/FreeMonoBold.otf", 32)

        for n, (im_orig, gt) in enumerate(zip(sample.images, sample.ground_truth)):
            # make a copy, so sample stays untouched:
            im = im_orig.copy()
            # draw GT and
            draw = ImageDraw.Draw(im)
            if gt and show_gt:
                gt2 = Rect(
                    gt.left - 1, gt.top - 1, gt.width + 2, gt.height + 2)
                draw.rectangle(gt.outer, None, 'green')
                draw.rectangle(gt2.outer, None, 'green')

            # draw prediction
            r = self.tracking_run.run[n]
            result = r['result']
            roi = r['roi']
            pos = result['predicted_position']
            roi2 = Rect(
                roi.left - 1, roi.top - 1, roi.width + 2, roi.height + 2)
            draw.rectangle(roi.outer, None, 'cyan')
            draw.rectangle(roi2.outer, None, 'cyan')
            pos2 = Rect(
                pos.left - 1, pos.top - 1, pos.width + 2, pos.height + 2)
            draw.rectangle(pos.outer, None, 'yellow')
            draw.rectangle(pos2.outer, None, 'yellow')

            # heatmap / mask
            heat = r['consolidation_images']['single']
            if n == 0:
                heat.save("/tmp/l1.png")
            draw = ImageDraw.Draw(heat)
            if gt and show_gt:
                draw.rectangle(
                    cc.capture_to_mask(gt, roi).outer, None, "green")
            draw.rectangle(cc.capture_to_mask(pos, roi).outer, None, "yellow")
            self.heats.append(heat)

            imout = im.copy()
            iw, ih = imout.size
            w, h = heat.size
            w *= 2
            h *= 2
            h2 = heat.copy().resize((w, h))
            draw = ImageDraw.Draw(imout)
            draw.rectangle(
                (iw - w - 2, ih - h - 2, iw, ih), "white", "white")
            imout.paste(h2, (iw - w, ih - h))
            draw.text((5, 0), "#%04d/%04d" %
                      (n + 1, l), "white", font=font)
            if result['updated'] == 'c':
                draw.text((iw - 130, 0), "dyn", "red", font=font)
            elif result['updated'] == 'f':
                draw.text((iw - 160, 0), "stat", "blue", font=font)

            dd = "/data/Peer/vis/out"
            imout.save(dd + "/%04d.jpg" % n)
#            imout.show()
#            exit()

            # sroi
            sroi = im_orig.crop(roi.outer).resize(cc.sroi_size)
            draw = ImageDraw.Draw(sroi)
            if gt and show_gt:
                draw.rectangle(
                    cc.capture_to_sroi(gt, roi).outer, None, "green")
            draw.rectangle(cc.capture_to_sroi(pos, roi).outer, None, "yellow")
            self.srois.append(sroi)

            t = "Frame #%04d/%04d" % (n + 1, sample.actual_frames)
            self.frames.append([im, t])

            # SGraphs:
            self.confidence_plotter.append(result['prediction_quality'])
            self.distance_plotter.append(result['center_distance'])
            self.overlap_plotter.append(result['overlap_score'])
            if result['updated'] == 'c':
                uc = 'red'
            elif result['updated'] == 'f':
                uc = 'blue'
            else:
                uc = None
            if uc:
                udraw.line(((n, 0), (n, 10)), uc, 1)
            print(pos, cc.capture_to_sroi(pos, roi),
                  cc.capture_to_mask(pos, roi))
        self.sample_text['text'] = "Sample %s/%s, Attributes: %s" % (
            sample.set_name, sample.name, ', '.join(sample.attributes))

        self.confidence = self.confidence_plotter.get_image()
        self.distance = self.distance_plotter.get_image()
        self.overlap = self.overlap_plotter.get_image()
        self.update = self.update_plotter

    def plot_frame(self, im, n):
        im = im.copy()
        for y in range(im.size[1]):
            r, g, b = im.getpixel((n, y))
            g = (g + 128) % 256
            im.putpixel((n, y), (r, g, b))
        return im

    def display_frame(self):
        if self.frames:
            n = self.current_frame
            (im, t) = self.frames[n]
            self.capture_image.set_image(im)
            self.video_text['text'] = t
            # heat
            self.consolidation_image.set_image(self.heats[n])
            # sroi
            self.sroi_image.set_image(self.srois[n])
            # plots
            self.confidence_plot.set_image(self.plot_frame(self.confidence, n))
            self.distance_plot.set_image(self.plot_frame(self.distance, n))
            self.overlap_plot.set_image(self.plot_frame(self.overlap, n))
            self.update_plot.set_image(self.plot_frame(self.update, n))

    def display_next_frame(self):
        self.current_frame += 1
        if self.current_frame >= len(self.frames):
            self.current_frame = 0
        self.display_frame()
        if self.running:
            self.root.after(self.wait_time, self.display_next_frame)

    def display_previous_frame(self):
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = len(self.frames) - 1
        self.display_frame()
        if self.running:
            self.root.after(self.wait_time, self.display_next_frame)

    def build_widgets(self):
        self.sample_text = tk.Label(self.root)
        self.sample_text.pack()
        self.texts['sample_text'] = ""  # self.sample_text
        self.video_text = tk.Label(self.root)
        self.video_text.pack()
        self.texts['video_text'] = self.video_text

        self.picture_frame = tk.Frame(self.root)
        self.picture_frame.pack(side=tk.TOP)

        self.capture_frame = tk.Frame(self.picture_frame)
        self.capture_frame.pack(side=tk.TOP)

        self.capture_image = ImageLabel(
            self.capture_frame, text="Capture", compound=tk.BOTTOM)
        self.capture_image.pack(side=tk.LEFT)
        self.images['capture_image'] = self.capture_image

        self.sroi_image = ImageLabel(
            self.capture_frame, text="SROI", compound=tk.BOTTOM)
        self.sroi_image.pack(side=tk.RIGHT)
        self.images['sroi_image'] = self.sroi_image

        self.consolidation_image = ImageLabel(self.picture_frame)
        self.consolidation_image.pack(side=tk.BOTTOM)
        self.images['consolidation_image'] = self.consolidation_image

        # ==
        self.figure_frame = tk.Frame(self.root)
        self.figure_frame.pack(side=tk.BOTTOM)

        self.confidence_plot = ImageLabel(
            self.figure_frame, text="Confidence", compound=tk.BOTTOM,)
        self.confidence_plot.pack()
        self.images['confidence_plot'] = self.confidence_plot

        self.distance_plot = ImageLabel(
            self.figure_frame, text="Distance", compound=tk.BOTTOM,)
        self.distance_plot.pack()
        self.images['distance_plot'] = self.distance_plot

        self.overlap_plot = ImageLabel(
            self.figure_frame, text="Overlap", compound=tk.BOTTOM,)
        self.overlap_plot.pack()
        self.images['overlap_plot'] = self.overlap_plot

        self.update_plot = ImageLabel(
            self.figure_frame, text="Update", compound=tk.BOTTOM)
        self.update_plot.pack()
        return

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

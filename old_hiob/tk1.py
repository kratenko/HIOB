"""
Created on 2016-11-03

@author: Peer Springstübe
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from hiob import data_set
from hiob.roi import SimpleRoiCalculator
from hiob.rect import Rect

path = "/data/Peer/data/tb100_unzipped/MotorRolling/img"


class App(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.roi_calculator = SimpleRoiCalculator()
        self.create_widgets()

    def create_dummy_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there['text'] = "Hallo World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")
        self.quit = tk.Button(
            self, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")

    def create_widgets(self):
        self.ds = data_set.load_tb100_sample("Jump")
        self.current = -1
        self.max = len(self.ds.images)
        self.delay = 1000 // 30

        self.capture_image = None
        self.capture_label = tk.Label(self)
        self.capture_label.pack()

        self.sroi_image = None
        self.sroi_label = tk.Label(self)
        self.sroi_label.pack()

        self.next_frame()

    def capture_to_sroi(self, pos, roi, sroi_size):
        """
        Convert rect in capture to rect in scaled roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = sroi_size[0] / rw
        scale_h = sroi_size[1] / rh
        ix = round((px - rx) * scale_w)
        iy = round((py - ry) * scale_h)
        iw = scale_w * pw
        ih = scale_h * ph
        return Rect(ix, iy, iw, ih)

    def sroi_to_capture(self, pos, roi, sroi_size):
        """
        Convert rect in scaled roi to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = sroi_size[0] / rw
        scale_h = sroi_size[1] / rh
        cx = round(sx / scale_w + rx)
        cy = round(sy / scale_h + ry)
        cw = sw / scale_w
        ch = sh / scale_h
        return Rect(cx, cy, cw, ch)

    def prepare_images(self):
        sroi_size = (368, 368)
        img = self.ds.images[self.current].copy()
        gt = self.ds.ground_truth[self.current]
        roi = self.roi_calculator.calculate_roi(img, gt)
        roi_gt = self.capture_to_sroi(gt, roi, sroi_size)
        back_gt = self.sroi_to_capture(roi_gt, roi, sroi_size)

        roi_img = img.crop(roi.outer).copy().resize(sroi_size)
        draw = ImageDraw.Draw(roi_img)
        draw.rectangle(roi_gt.outer, None, (255, 255, 255, 255))

        draw = ImageDraw.Draw(img)
        draw.rectangle(gt.outer, None, (255, 255, 255, 255))
        draw.rectangle(back_gt.outer, None, (255, 0, 255, 255))
        draw.rectangle(roi.outer, None, (0, 255, 255, 255))
        self.capture_image = ImageTk.PhotoImage(img)
        self.sroi_image = ImageTk.PhotoImage(roi_img)

    def next_frame(self):
        # go to next frame:
        self.current += 1
        if self.current >= self.max:
            self.current = 0
        # prepare all images to be displayed
        self.prepare_images()
        self.capture_label['image'] = self.capture_image
        self.sroi_label['image'] = self.sroi_image
        self.after(self.delay, self.next_frame)

root = tk.Tk()
app = App(master=root)
app.mainloop()

import os, sys, io, re
import logging
import zipfile
from PIL import Image
import re


class DataSample(object):
    def __init__(self):
        pass
    
    def _load_image_from_zip(self, zf, path):
        data = zf.read(path)
        stream = io.BytesIO(data)
        im = Image.open(stream)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im
    
    def _extract_name_from_zip(self, zf):
        for fn in zf.namelist():
            if fn.endswith('/img/'):
                return fn[:-5]
        return None
    
    def _extract_image_paths_from_zip(self, zf):
        paths = []
        r = re.compile(r".+/img/\d+\.[0-9a-zA-Z]+")
        for fn in zf.namelist():
            if r.match(fn):
                paths.append(fn)
        return sorted(paths)
    
    def _extract_groundtruth_from_zip(self, zf):    
        paths = []
        r = re.compile(r".+/groundtruth_rect.*\.txt")
        for fn in zf.namelist():
            if r.match(fn):
                paths.append(fn)
        return sorted(paths)
        
    def _load_groundtruth_from_zip(self, zf, path):
        r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
        rects = []
        with zf.open(path, 'r') as tf:
            for line in tf:
                m = r.match(line)
                if m:
                    rect = tuple(int(m.group(n)) for n in (1,2,3,4))
                else:
                    rect = None
                rects.append(rect)
        return rects  
                
    
    def load_from_zip(self, path):
        self.path = path
        with zipfile.ZipFile(path, 'r') as zf:
            # extract info from filenames within zip:
            self.name = self._extract_name_from_zip(zf)
            self.image_paths = self._extract_image_paths_from_zip(zf)
            self.groundtruth_paths = self._extract_groundtruth_from_zip(zf)
            # load images:
            self.frames = [self._load_image_from_zip(zf, image_path) for image_path in self.image_paths]
            # load truths
            self.groundtruths = [self._load_groundtruth_from_zip(zf, truth_path) for truth_path in self.groundtruth_paths]
        
    # --
    
    def _load_princeton_image(self, path):
        with open(path) as image_file:
            im = Image.open(image_file)
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
        return im
    
    def _load_princeton_gt(self, path):
        _, name = os.path.split(path)
        gt_path = os.path.join(path, name+".txt")
        if os.path.isfile(gt_path):
            gt = []
            with open(gt_path) as input_file:
                for line in input_file:
                    (x, y, w, h, _) = line.split(",")
                    if x == 'NaN':
                        gt.append(None)
                    else:
                        gt.append([int(x), int(y), int(w), int(h)])
        return gt
    
    def load_princeton(self, path):
        self.path = path
        image_dir = os.path.join(path, 'rgb')
        image_paths = os.listdir(image_dir)
        r = re.compile(r"^r-(\d+)-(\d+).(\w+)$")
        d = {}
        for p in image_paths:
            m = r.match(p)
            if m:
                (_, num, _) = m.groups()
                d[int(num)] = p
        file_names = [d[k] for k in sorted(d.keys())]
        self.frames = [self._load_princeton_image(os.path.join(image_dir, file_name)) for file_name in file_names]
        self.groundtruths = [self._load_princeton_gt(path)]

        

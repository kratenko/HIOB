import logging
import transitions
from PIL import ImageTk, Image
import queue
import threading
import argparse
import collections
from PIL.ImageDraw import Draw

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)

import tkinter as tk


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

    def __init__(self, conf):
        self.conf = conf

        self.root = tk.Tk()
        self.root.title("Hiob")

        self.dead = False
        self.queue = queue.Queue()
        self.images = {}
        self.texts = {}
        self.build_widgets()

        logger.info("starting tracker")
        self.start_tracker()
        logger.info("starting consumer for queue")
        self.consume_loop()

    def build_widgets(self):
        self.sample_text = tk.Label(self.root)
        self.sample_text.pack()
        self.texts['sample_text'] = self.sample_text
        self.video_text = tk.Label(self.root)
        self.video_text.pack()
        self.texts['video_text'] = self.video_text

        self.capture_frame = tk.Frame(self.root)
        self.capture_frame.pack()

        self.capture_image = ImageLabel(
            self.capture_frame, text="Capture", compound=tk.BOTTOM)
        self.capture_image.pack(side=tk.LEFT)
        self.images['capture_image'] = self.capture_image

        self.sroi_image = ImageLabel(
            self.capture_frame, text="SROI", compound=tk.BOTTOM)
        self.sroi_image.pack(side=tk.RIGHT)
        self.images['sroi_image'] = self.sroi_image

        self.consolidation_image = ImageLabel(self.root)
        self.consolidation_image.pack()
        self.images['consolidation_image'] = self.consolidation_image

        self.figure_frame = tk.Frame(self.root)
        self.figure_frame.pack()

        self.center_distance_figure = ImageLabel(self.figure_frame)
        self.center_distance_figure.pack(side=tk.LEFT)
        self.images['center_distance_figure'] = self.center_distance_figure

        self.overlap_score_figure = ImageLabel(self.figure_frame)
        self.overlap_score_figure.pack(side=tk.RIGHT)
        self.images['overlap_score_figure'] = self.overlap_score_figure

        self.lost_figure = ImageLabel(self.figure_frame)
        self.lost_figure.pack(side=tk.RIGHT)
        self.images['lost_figure'] = self.lost_figure
#
        self.confidence_plotter = SGraph(length=100)
        self.confidence_plot = ImageLabel(self.figure_frame)
        self.confidence_plot.pack()
        self.images['confidence_plot'] = self.confidence_plot

        self.confidence_plotter = SGraph(
            min_y=0, max_y=1.0, length=100, height=100)
        self.confidence_plot = ImageLabel(
            self.figure_frame, text="Confidence", compound=tk.BOTTOM,)
        self.confidence_plot.pack(side=tk.LEFT)
        self.images['confidence_plot'] = self.confidence_plot

        self.distance_plotter = SGraph(
            min_y=0, max_y=100, length=100, height=100)
        self.distance_plotter.ylines = [20]
        self.distance_plot = ImageLabel(
            self.figure_frame, text="Distance", compound=tk.BOTTOM,)
        self.distance_plot.pack(side=tk.LEFT)
        self.images['distance_plot'] = self.distance_plot

        self.overlap_plotter = SGraph(
            min_y=0, max_y=1.0, length=100, height=100)
        self.overlap_plot = ImageLabel(
            self.figure_frame, text="Overlap", compound=tk.BOTTOM,)
        self.overlap_plot.pack(side=tk.LEFT)
        self.images['overlap_plot'] = self.overlap_plot

        self.lost_plotter = SGraph(
            min_y=0.0, max_y=3.0, length=100, height=100)
        self.lost_plot = ImageLabel(
            self.figure_frame, text="Lost", compound=tk.BOTTOM,)
        self.lost_plot.pack(side=tk.LEFT)
        self.images['lost_plot'] = self.lost_plot

    def consume_entry(self, entry):
        for k, v in entry.items():
            if k in self.images:
                self.images[k].set_image(v)
            elif k in self.texts:
                self.texts[k]['text'] = v

    def consume_loop(self):
        while True:
            try:
                entry = self.queue.get_nowait()
                self.consume_entry(entry)
            except queue.Empty:
                break
        self.root.after(10, self.consume_loop)

    def feed_queue(self, entry):
        self.queue.put(entry)

    def start_tracker(self):
        self.tracker_thread = threading.Thread(target=self.tracker_fun)
        self.tracker_thread.start()

    def verify_running(self):
        if self.dead:
            raise AppTerminated()

    def tracker_one(self, tracker, sample):
        tracking = tracker.start_tracking_sample(
            sample)

        # feature selection:
        tracking.start_feature_selection()
        sample = tracking.sample
        self.feed_queue(
            {'sroi_image': tracking.get_frame_sroi_image(decorations=True),
             'capture_image': tracking.get_frame_capture_image(),
             'sample_text': "Sample %s/%s, Attributes: %s" % (
                sample.set_name, sample.name, ', '.join(sample.attributes)),
             'video_text': "Frame #%04d/%04d" % (1, sample.actual_frames),
             })
        while not tracking.feature_selection_done():
            self.verify_running()
            tracking.feature_selection_step()
        tracking.finish_feature_selection()

        # consolidator training:
        tracking.start_consolidator_training()
        while not tracking.consolidator_training_done():
            self.verify_running()
            tracking.consolidator_training_step()
            logger.info("COST: %f", tracking.consolidator_training_cost())
        tracking.finish_consolidator_training()

        # add threshold lines to confidence plotter:
        confidence_lines = []
        if tracking.tracker.consolidator.update_threshold:
            confidence_lines.append(
                tracking.tracker.consolidator.update_threshold)
        if tracking.tracker.consolidator.update_lower_threshold:
            confidence_lines.append(
                tracking.tracker.consolidator.update_lower_threshold)
        self.confidence_plotter.ylines = confidence_lines

        # tracking:
        tracking.start_tracking()
        while tracking.frames_left():
            self.verify_running()
            tracking.tracking_step()
#            evs = tracking.get_evaluation_figures()
            sample = tracking.sample
            cf = tracking.current_frame_number
            fr = tracking.current_frame.result
            self.confidence_plotter.append(
                tracking.current_frame.prediction_quality)
            self.distance_plotter.append(fr['center_distance'])
            self.overlap_plotter.append(fr['overlap_score'])
            self.lost_plotter.append(fr['lost'])
            entry = {
                'capture_image': tracking.get_frame_capture_image(),
                'sroi_image': tracking.get_frame_sroi_image(),
                'sample_text': "Sample %s/%s, Attributes: %s" % (
                    sample.set_name, sample.name, ', '.join(sample.attributes)),
                'video_text': "Frame #%04d/%04d" % (cf, sample.actual_frames),
                'consolidation_image': tracking.get_frame_consolidation_images()['single'],
                #                'center_distance_figure': evs['center_distance'],
                #                'overlap_score_figure': evs['overlap_score'],
                'confidence_plot': self.confidence_plotter.get_image(),
                'distance_plot': self.distance_plotter.get_image(),
                'overlap_plot': self.overlap_plotter.get_image(),
                'lost_plot': self.lost_plotter.get_image(),
            }
            self.feed_queue(entry)
        tracking.finish_tracking()

        return tracking

    def tracker_fun(self):
        from hiob.tracker import Tracker
        tracker = Tracker(self.conf)
        tracker.setup_environment()
        try:
            with tracker.setup_session():
                for sample in tracker.samples:
                    sample.load()
                    tracking = self.tracker_one(tracker, sample)
                    tracker.evaluate_tracking(tracking)
                    sample.unload()
                tracker.evaluate_tracker()
        except AppTerminated:
            logger.info("App terminated, ending tracker thread early.")
            # tracking.execute_consolidator_training()
            # tracking.execute_tracking()

        logger.info("Leaving tracker thread")

    def run(self):
        self.root.mainloop()
        self.dead = True
        if self.tracker_thread:
            self.tracker_thread.join()

if __name__ == '__main__':
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment')
    parser.add_argument('-t', '--tracker')
    args = parser.parse_args()

    # create Configurator
    from hiob.configuration import Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=args.environment,
        tracker_path=args.tracker,
    )

    # execute gui app and run tracking
    logger.info("Initiate tracking process in gui app")
    app = App(conf)
    app.run()

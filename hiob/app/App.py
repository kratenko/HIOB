import queue
import threading
import tkinter as tk
import asyncio

from .AppTerminatedException import AppTerminatedException
from .ImageLabel import ImageLabel
from .SGraph import SGraph
from Tracker import Tracker


class App:

    def __init__(self, logger, conf):
        self.conf = conf

        self.root = tk.Tk()
        self.root.title("Hiob")

        self.dead = False
        self.queue = queue.Queue()
        self.images = {}
        self.texts = {}
        self.build_widgets()
        self.logger = logger

        self.logger.info("starting tracker")
        self.start_tracker()
        self.logger.info("starting consumer for queue")
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

        self.relative_center_distance_figure = ImageLabel(self.figure_frame)
        self.relative_center_distance_figure.pack(side=tk.LEFT)
        self.images['relative_center_distance_figure'] = self.relative_center_distance_figure

        self.overlap_score_figure = ImageLabel(self.figure_frame)
        self.overlap_score_figure.pack(side=tk.RIGHT)
        self.images['overlap_score_figure'] = self.overlap_score_figure

        self.adjusted_overlap_score_figure = ImageLabel(self.figure_frame)
        self.adjusted_overlap_score_figure.pack(side=tk.RIGHT)
        self.images['adjusted_overlap_score_figure'] = self.adjusted_overlap_score_figure

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

        self.relative_distance_plotter = SGraph(
            min_y=0, max_y=100, length=100, height=100)
        self.relative_distance_plotter.ylines = [20]
        self.relative_distance_plot = ImageLabel(
            self.figure_frame, text="Rel. Distance", compound=tk.BOTTOM,)
        self.relative_distance_plot.pack(side=tk.LEFT)
        self.images['relative_distance_plot'] = self.relative_distance_plot

        self.overlap_plotter = SGraph(
            min_y=0, max_y=1.0, length=100, height=100)
        self.overlap_plot = ImageLabel(
            self.figure_frame, text="Overlap", compound=tk.BOTTOM,)
        self.overlap_plot.pack(side=tk.LEFT)
        self.images['overlap_plot'] = self.overlap_plot

        self.adjusted_overlap_plotter = SGraph(
            min_y=0, max_y=1.0, length=100, height=100)
        self.adjusted_overlap_plot = ImageLabel(
            self.figure_frame, text="Adjusted Overlap", compound=tk.BOTTOM,)
        self.adjusted_overlap_plot.pack(side=tk.LEFT)
        self.images['adjusted_overlap_plot'] = self.adjusted_overlap_plot

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
            raise AppTerminatedException()

    async def tracker_one(self, tracker, sample):

        tracking = await tracker.start_tracking_sample(
            sample)

        # feature selection:
        tracking.start_feature_selection()
        sample = tracking.sample
        self.feed_queue(
            {'sroi_image': tracking.get_frame_sroi_image(decorations=True),
             'capture_image': tracking.get_frame_capture_image(),
             'sample_text': "Sample %s/%s, Attributes: %s" % (
                sample.set_name, sample.name, ', '.join(sample.attributes)),
             'video_text': "Frame #%04d/%04d" % (sample.current_frame_id, sample.actual_frames),
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
            self.logger.info("COST: %f", tracking.consolidator_training_cost())
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
            await tracking.tracking_step()
#            evs = tracking.get_evaluation_figures()
            sample = tracking.sample
            cf = tracking.current_frame_number
            fr = tracking.current_frame.result
            self.confidence_plotter.append(
                tracking.current_frame.prediction_quality)
            self.distance_plotter.append(fr['center_distance'])
            self.relative_distance_plotter.append(fr['relative_center_distance'])
            self.overlap_plotter.append(fr['overlap_score'])
            self.adjusted_overlap_plotter.append(fr['adjusted_overlap_score'])
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
                'relative_distance_plot': self.relative_distance_plotter.get_image(),
                'overlap_plot': self.overlap_plotter.get_image(),
                'adjusted_overlap_plot': self.adjusted_overlap_plotter.get_image(),
                'lost_plot': self.lost_plotter.get_image(),
            }
            self.feed_queue(entry)
        tracking.finish_tracking()

        return tracking

    def tracker_fun(self):
        tracker = Tracker(self.conf)
        tracker.setup_environment()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with tracker.setup_session():
                for sample in tracker.samples:
                    sample.load()
                    tracking = loop.run_until_complete(self.tracker_one(tracker, sample))
                    tracker.evaluate_tracking(tracking)
                    sample.unload()
                tracker.evaluate_tracker()
        except AppTerminatedException:
            self.logger.info("App terminated, ending tracker thread early.")
            # tracking.execute_consolidator_training()
            # tracking.execute_tracking()
        loop.close()

        self.logger.info("Leaving tracker thread")

    def run(self):
        self.root.mainloop()
        self.dead = True
        if self.tracker_thread:
            self.tracker_thread.join()
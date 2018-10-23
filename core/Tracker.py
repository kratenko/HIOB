import datetime
import errno
import logging
import os
import random
import shutil
import socket
import uuid
import re
import signal

import numpy as np
import tensorflow as tf

from . import Consolidator
from . import evaluation
from . import extraction
from . import pursuing
from . import roi
from . import selection
from .sample_provider import DataDirectory
from .Tracking import Tracking

logger = logging.getLogger(__name__)


class Tracker:
    def _find_out_git_revision(self):
        import git
        try:
            repo = git.Repo(search_parent_directories=False)
            self.git_revision = repo.head.object.hexsha
            self.git_dirty = repo.is_dirty()
        except git.exc.InvalidGitRepositoryError:
            self.git_revision = "--INVALID--"
            self.git_dirty = True

    def abort(self, signum, _):
        self.interrupt_received = True
        if self.current_sample is not None:
            self.current_sample.unload()
        print("received abort signal! Exiting")

    def __init__(self, configuration):
        signal.signal(signal.SIGINT, self.abort)
        signal.signal(signal.SIGTERM, self.abort)
        signal.signal(signal.SIGQUIT, self.abort)
        signal.signal(signal.SIGABRT, self.abort)

        logger.warning("CREATING NEW TRACKER")
        self.context = None
        self.configuration = configuration
        self.interrupt_received = False

        # prepare random seeds:
        self.py_seed = None
        if 'random_seed' in configuration:
            self.py_seed = configuration['random_seed']
        if self.py_seed is None:
            logger.info("No random seed given, creating one from random")
            self.py_seed = random.randint(0, 0xffffffff)
        logger.info(
            "Master Random Seed is {0} = 0x{1}".format(self.py_seed, self.py_seed))
        self.py_random = random.Random(self.py_seed)
        self.np_seed = self.py_random.randint(0, 0xffffffff)
        self.np_random = np.random.RandomState(self.np_seed)
        self.tf_seed = self.py_random.randint(0, 0xffffffff)
        self.configuration.set_override('py_random', self.py_random)
        self.configuration.set_override('np_random', self.np_random)
        tf.set_random_seed(self.tf_seed)

        # find out exact version:
        self._find_out_git_revision()

        self.ts_created = datetime.datetime.now()
        self.ts_done = None

        self.log_dir = configuration['log_dir']
        self.data_dir = re.sub(r"[/\\]", os.path.sep.replace("\\", "\\\\"), configuration['data_dir'])
        #print(configuration["data_dir"], "->", self.data_dir)
        self.configuration.set_override("tracking", [re.sub(r"[/\\]", os.path.sep.replace("\\", "\\\\"), name)
                                                     for name in self.configuration["tracking"]])
        #print(tracking_tmp, "->", configuration["tracking"][0])
        self.sroi_size = configuration['sroi_size']

        print("tracking_conf:")
        print("fake_fps: {}\nskip_frames: {}\nshuffle: {}".format(
            configuration["tracking_conf"]["fake_fps"],
            configuration["tracking_conf"]["skip_frames"],
            configuration["tracking_conf"]["shuffle"]))

        self.modules = []
        self.session = None

        #
        self.total_center_distances = np.empty(0)
        self.total_relative_center_distances = np.empty(0)
        self.total_overlap_scores = np.empty(0)
        self.total_adjusted_overlap_scores = np.empty(0)
        self.tracking_evaluations = []

        # samples to track
        self.data_directory = DataDirectory(data_dir=self.data_dir)
        self.samples = []
        if 'tracking' in self.configuration:
            self.samples = self.data_directory.evaluate_sample_list(
                self.configuration['tracking'], self.configuration['tracking_conf'])
            if "shuffle" in self.configuration['tracking_conf'] and self.configuration['tracking_conf']['shuffle']:
                random.shuffle(self.samples)

        self.roi_calculator = roi.SimpleRoiCalculator()
        self.modules.append(self.roi_calculator)
        self.sroi_generator = roi.SimpleSroiGenerator()
        self.modules.append(self.sroi_generator)
        self.feature_extractor = extraction.CnnFeatureExtractor()
        self.modules.append(self.feature_extractor)
        self.feature_selector = selection.NetSelector()
        self.modules.append(self.feature_selector)
        self.consolidator = Consolidator.SingleNetConsolidator()
        self.modules.append(self.consolidator)
        self.pursuer = pursuing.SwarmPursuer()
        self.modules.append(self.pursuer)

        # configure modules
        self.roi_calculator.configure(configuration)
        self.sroi_generator.configure(configuration)
        self.feature_extractor.configure(configuration)
        self.feature_selector.configure(configuration)
        self.consolidator.configure(configuration)
        self.pursuer.configure(configuration)
        self.is_setup = False

        self.current_sample = None

    def setup_environment(self):
        logger.info("Setting up environment")
        self.log_formatter = logging.Formatter(
            '[%(asctime)s|%(levelname)s|%(name)s|%(filename)s:%(lineno)d] - %(message)s')

        self.log_console_handler = logging.StreamHandler()
        self.log_console_handler.setFormatter(self.log_formatter)
        self.log_console_handler.setLevel(self.configuration['log_level'])

        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.log_console_handler)

        if self.configuration['log_level'] == logging.INFO:
            self.priority_log_console_handler = self.log_console_handler
            self.logging_context_manager = LoggingContextManager(None)
        else:
            self.priority_log_console_handler = logging.StreamHandler()
            self.priority_log_console_handler.setFormatter(self.log_formatter)
            self.priority_log_console_handler.setLevel(logging.INFO)
            logger.addHandler(self.priority_log_console_handler)
            self.logging_context_manager = LoggingContextManager(self.priority_log_console_handler)



        # base dir for hiob operations:
        logger.info("log_dir is '%s'", self.log_dir)
        try:
            os.makedirs(self.log_dir)
        except OSError as exception:
            if exception.errno == errno.EEXIST:
                logger.info("log_dir exists")
            else:
                logger.error(
                    "Failed to create log_dir at '%s'", self.log_dir)
                raise
        else:
            logger.info("log_dir did not exist, created log_dir")
        # name for this run:
        self.execution_host = socket.gethostname()
        self.execution_name = (
            'hiob-execution-{}-{:%Y-%m-%d-%H.%M.%S.%f}'.format(self.execution_host, self.ts_created))
        # uuid for this run:
        self.execution_id = str(uuid.uuid4())
        logger.info(
            "execution: " + self.execution_id + ' - ' + self.execution_name)
        self.environment_name = self.configuration['environment_name']
        # create execution dir:
        self.execution_dir = os.path.join(self.log_dir, self.execution_name)
        logger.info("creating execution_dir at '%s'", self.execution_dir)
        os.makedirs(self.execution_dir)
        # switch logging to log to file:
        self.log_file = os.path.join(self.execution_dir, 'execution.log')
        logger.info(
            "Creating file handler for logging into '%s'", self.log_file)
        self.log_file_handler = logging.FileHandler(self.log_file)
        self.log_file_handler.setFormatter(self.log_formatter)
        self.root_logger.addHandler(self.log_file_handler)
        logger.info("Attached log to file '%s'", self.log_file)
        logger.info(
            "Execution is: %s - '%s'", self.execution_id, self.execution_name)
        logger.info("log_dir: '%s'", self.log_dir)
        logger.info("execution_dir: '%s'", self.execution_dir)
        # copy config files:
        logger.info("Copying environment file from '%s'",
                    self.configuration.environment_path)
        shutil.copyfile(
            self.configuration.environment_path, os.path.join(self.execution_dir, 'environment.yaml'))
        logger.info(
            "Copying tracker file from '%s'", self.configuration.tracker_path)
        shutil.copyfile(self.configuration.tracker_path, os.path.join(
            self.execution_dir, 'tracker.yaml'))

    def setup(self, sample):
        self.current_sample = sample
        self.current_sample.load(self.logging_context_manager)
        self.setup_session()

        if self.is_setup and self.context is not None \
                and (self.current_sample is None or sample.capture_size != self.current_sample.capture_size):
            return self.context

        #if self.current_sample is not None:
        #self.current_sample = sample

        # setup modules
        self.roi_calculator.setup(self)
        size = sample.capture_size
        #size = size[1], size[0]
        self.sroi_generator.setup(self, size)
        self.feature_extractor.setup(self, self.sroi_generator.generated_sroi)
        # cannot know mask size up to this point:
        self.mask_size = self.feature_extractor.output_size
        self.configuration.set_override('mask_size', self.mask_size)
        self.configuration.set_override(
            'output_features', self.feature_extractor.output_features)
        self.feature_selector.setup(
            self)
        self.consolidator.setup(self)
        self.pursuer.setup(self)
        self.is_setup = True
        logger.info("Setup done")

        return self.context

    def setup_session(self):
        self.context = TrackerContext(self)
        return self.context
        #self.setup(sample)

    async def start_tracking_sample_by_name(self, set_name, sample_name):
        sample = self.data_directory.get_sample(set_name, sample_name)
        #self.session = tf.Session()
        #self.setup(sample)
        return await self.start_tracking_sample(sample)

    async def start_tracking_sample(self, sample):
        #self.setup(sample)
        logging.info("start tracking sample {}".format(sample.name))
        tracking = Tracking(tracker=self, session=self.session)

        # setup tracking:
        self.feature_selector.setup_tracking(
            tracking.module_states.feature_selector, self.feature_extractor.output_features)
        self.consolidator.setup_tracking(
            tracking.module_states.consolidator, self.feature_extractor.output_features)

        # TODO: this should not be done here! maybe initialise only new
        # TODO:: variables?
        # self.session.run(tf.initialize_all_variables())
        self.session.run(tf.global_variables_initializer())

        await tracking.load_sample(sample)
        return tracking

    def evaluate_tracking(self, tracking):
        self.tracking_evaluations.append(tracking.evaluation)

        sample_frames = tracking.get_total_frames()
        assert len(tracking.tracking_log) == sample_frames
        center_distances = np.empty(sample_frames)
        relative_center_distances = np.empty(sample_frames)
        overlap_scores = np.empty(sample_frames)
        adjusted_overlap_scores = np.empty(sample_frames)
        for n, l in enumerate(tracking.tracking_log):
            center_distances[n] = l['result']['center_distance']
            relative_center_distances[n] = l['result']['relative_center_distance']
            overlap_scores[n] = l['result']['overlap_score']
            adjusted_overlap_scores[n] = l['result']['adjusted_overlap_score']
        # store for total
        self.total_center_distances = np.append(
            self.total_center_distances, center_distances)
        self.total_relative_center_distances = np.append(
            self.total_relative_center_distances, relative_center_distances)
        self.total_overlap_scores = np.append(
            self.total_overlap_scores, overlap_scores)
        self.total_adjusted_overlap_scores = np.append(
            self.total_adjusted_overlap_scores, adjusted_overlap_scores)

        evaluation.print_tracking_evaluation(tracking.evaluation, self.logging_context_manager)

    async def execute_tracking_on_sample(self, sample):

        with self.setup(sample):
            #sample.load(self.logging_context_manager)
            if not self.is_setup:
                self.setup(sample)
            tracking = await self.start_tracking_sample(sample)
            await tracking.execute_everything()
            self.evaluate_tracking(tracking)
            #sample.unload()

    async def execute_tracking_on_all_samples(self):
        for sample in self.samples:
            #with self.setup(sample):
            await self.execute_tracking_on_sample(sample)

    async def execute_everything(self):
        await self.execute_tracking_on_all_samples()
        self.evaluate_tracker()

    def evaluate_tracker(self):
        self.ts_done = datetime.datetime.now()
        self.evaluation = evaluation.do_tracker_evaluation(self)


class LoggingContextManager:
    def __init__(self, log_handler):
        self.log_handler = log_handler

    def __call__(self, logger):
        return ForceLoggingContext(logger, self.log_handler)


class ForceLoggingContext:
    def __init__(self, logger, log_handler):
        self.logger = logger
        self.log_handler = log_handler

    def __enter__(self):
        if self.log_handler is not None:
            self.logger.addHandler(self.log_handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_handler is not None:
            self.logger.removeHandler(self.log_handler)


class TrackerContext:
    def __init__(self, tracker):
        self.tracker = tracker

    def __enter__(self):
        #self.tracker.current_sample.load()
        print("setting up new session")
        self.tracker.session = tf.Session()
        return self.tracker.session.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("closing session...")
        self.tracker.session.__exit__(exc_type, exc_val, exc_tb)
        self.tracker.current_sample.unload()
        tf.reset_default_graph()

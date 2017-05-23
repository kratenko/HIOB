import logging
import os
import uuid
import errno
import datetime
import socket
import shutil
import random

import numpy as np
import tensorflow as tf

import hiob.roi
import hiob.extraction
import hiob.selection
import hiob.consolidation
import hiob.pursuing
from hiob.tracking import Tracking
from hiob import evaluation
from hiob.data_set import DataDirectory

logger = logging.getLogger(__name__)


class Tracker(object):

    def _find_out_git_revision(self):
        import git
        repo = git.Repo(search_parent_directories=True)
        self.git_revision = repo.head.object.hexsha
        self.git_dirty = repo.is_dirty()

    def __init__(self, configuration):
        self.configuration = configuration

        # prepare random seeds:
        self.py_seed = None
        if 'random_seed' in configuration:
            self.py_seed = configuration['random_seed']
        if self.py_seed is None:
            logger.info("No random seed given, creating one from random")
            self.py_seed = random.randint(0, 0xffffffff)
        logger.info(
            "Master Random Seed is %d = 0x%h", self.py_seed, self.py_seed)
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
        self.data_dir = configuration['data_dir']
        self.sroi_size = configuration['sroi_size']

        self.modules = []
        self.session = None

        #
        self.total_center_distances = np.empty(0)
        self.total_overlap_scores = np.empty(0)
        self.tracking_evaluations = []

        # samples to track
        self.data_directory = DataDirectory(data_dir=self.data_dir)
        self.samples = []
        if 'tracking' in self.configuration:
            self.samples = self.data_directory.evaluate_sample_list(
                self.configuration['tracking'])

        self.roi_calculator = hiob.roi.SimpleRoiCalculator()
        self.modules.append(self.roi_calculator)
        self.sroi_generator = hiob.roi.SimpleSroiGenerator()
        self.modules.append(self.sroi_generator)
        self.feature_extractor = hiob.extraction.CnnFeatureExtractor()
        self.modules.append(self.feature_extractor)
        self.feature_selector = hiob.selection.NetSelector()
        self.modules.append(self.feature_selector)
        self.consolidator = hiob.consolidation.SingleNetConsolidator()
        self.modules.append(self.consolidator)
        self.pursuer = hiob.pursuing.SwarmPursuer()
        self.modules.append(self.pursuer)

        # configure modules
        self.roi_calculator.configure(configuration)
        self.sroi_generator.configure(configuration)
        self.feature_extractor.configure(configuration)
        self.feature_selector.configure(configuration)
        self.consolidator.configure(configuration)
        self.pursuer.configure(configuration)

    def setup_environment(self):
        logger.info("Setting up environment")
        self.log_formatter = logging.Formatter(
            '[%(asctime)s|%(levelname)s|%(name)s|%(filename)s:%(lineno)d] - %(message)s')

        self.log_console_handler = logging.StreamHandler()
        self.log_console_handler.setFormatter(self.log_formatter)
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.log_console_handler)

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

    def setup(self, session):
        self.session = session
        # setup modules
        self.roi_calculator.setup(self.session)
        self.sroi_generator.setup(self.session)
        self.feature_extractor.setup(self.session)
        # cannot know mask size up to this point:
        self.mask_size = self.feature_extractor.output_size
        self.configuration.set_override('mask_size', self.mask_size)
        self.configuration.set_override(
            'output_features', self.feature_extractor.output_features)
        self.feature_selector.setup(
            self.session)
        self.consolidator.setup(self.session)
        self.pursuer.setup(self.session)

    def setup_session(self):
        self.setup(tf.Session())
        return self.session

    def start_tracking_sample_by_name(self, set_name, sample_name):
        sample = self.data_directory.get_sample(set_name, sample_name)
        return self.start_tracking_sample(sample)

    def start_tracking_sample(self, sample):
        tracking = Tracking(tracker=self)

        # setup tracking:
        self.feature_selector.setup_tracking(
            tracking.module_states.feature_selector, self.feature_extractor.output_features)
        self.consolidator.setup_tracking(
            tracking.module_states.consolidator, self.feature_extractor.output_features)

        # TODO: this should not be done here! maybe initialise only new
        # TODO:: variables?
        # self.session.run(tf.initialize_all_variables())
        self.session.run(tf.global_variables_initializer())

        tracking.load_sample(sample)
        return tracking

    def evaluate_tracking(self, tracking):
        self.tracking_evaluations.append(tracking.evaluation)

        sample_frames = tracking.total_frames
        assert len(tracking.tracking_log) == sample_frames
        center_distances = np.empty(sample_frames)
        overlap_scores = np.empty(sample_frames)
        for n, l in enumerate(tracking.tracking_log):
            center_distances[n] = l['result']['center_distance']
            overlap_scores[n] = l['result']['overlap_score']
        # store for total
        self.total_center_distances = np.append(
            self.total_center_distances, center_distances)
        self.total_overlap_scores = np.append(
            self.total_overlap_scores, overlap_scores)

    def execute_tracking_on_sample(self, sample):
        tracking = self.start_tracking_sample(sample)
        tracking.execute_everything()
        self.evaluate_tracking(tracking)

    def execute_tracking_on_all_samples(self):
        for sample in self.samples:
            sample.load()
            self.execute_tracking_on_sample(sample)
            sample.unload()

    def execute_everything(self):
        self.execute_tracking_on_all_samples()
        self.evaluate_tracker()

    def evaluate_tracker(self):
        self.ts_done = datetime.datetime.now()
        self.evaluation = evaluation.do_tracker_evaluation(self)

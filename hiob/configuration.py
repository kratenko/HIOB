"""
Created on 2016-11-29

@author: Peer Springstübe
"""
import os
import yaml
import logging
import sys

logger = logging.getLogger(__name__)


class Configurator(object):

    def __init__(self, environment_path=None, tracker_path=None):
        logger.info("Building Configurator")
        if environment_path is None:
            self.environment_path = os.path.join('.', 'environment.yaml')
        else:
            self.environment_path = environment_path
        if tracker_path is None:
            self.tracker_path = os.path.join('.', 'tracker.yaml')
        else:
            self.tracker_path = tracker_path
        self.overrides = {}
        self.load_files()

    def load_files(self):
        try:
            logger.info(
                "Loading environment file from '%s'", self.environment_path)
            with open(self.environment_path, 'r') as f:
                self.environment = yaml.safe_load(f)
        except IOError as e:
            msg = "Could not open environment configuration file: %s", e
            logger.error(msg)
            print(msg, file=sys.stderr)
            exit(1)
        try:
            logger.info("Loading tracker file from '%s'", self.tracker_path)
            with open(self.tracker_path, 'r') as f:
                self.tracker = yaml.safe_load(f)
        except IOError as e:
            msg = "Could not open tracker configuration file: %s", e
            logger.error(msg)
            print(msg, file=sys.stderr)
            exit(1)

    def __getitem__(self, key):
        if key in self.overrides:
            return self.overrides[key]
        elif key in self.tracker:
            return self.tracker[key]
        elif key in self.environment:
            return self.environment[key]
        else:
            raise KeyError(key)

    def __contains__(self, key):
        if key in self.overrides:
            return True
        elif key in self.tracker:
            return True
        elif key in self.environment:
            return True
        else:
            return False

    def set_override(self, key, value):
        self.overrides[key] = value

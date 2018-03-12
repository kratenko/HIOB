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
            self.environment_path = os.path.join('.', 'config', 'environment.yaml')
        else:
            self.environment_path = environment_path
        if tracker_path is None:
            self.tracker_path = os.path.join('.', 'config', 'tracker.yaml')
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

        self.set_override("ros_mode", False)
        for sname in self['tracking']:
            p1, p2 = sname.split(os.path.sep, 1)
            if p1 == "ros":
                print("Found live sample; enabling ros mode.")
                self.set_override("ros_mode", True)

        if self['ros_mode'] and len(self['tracking']) > 1:
            print("Ros mode is enabled, but multiple samples have been defined. Disabling all but the first ros sample.")
            self.set_override('tracking', [list(filter(lambda x: len(x) >= 3 and x[:4] == 'ros/', self['tracking']))[0]])

        print("-----------------------------------------------------------------------------")
        print("ros mode is {0}.".format("TRUE" if self["ros_mode"] else "FALSE"))
        print("-----------------------------------------------------------------------------")

        if self['ros_mode']:
            import rospkg
            rp = rospkg.RosPack()
            for key in self.environment:
                self.environment[key] = self.environment[key].replace("\\share\\", rp.get_path('hiob_ros'))

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

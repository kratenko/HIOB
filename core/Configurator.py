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

    def __init__(self, hiob_path=None, environment_path=None, tracker_path=None, ros_config=None, silent=False):
        logger.info("Building Configurator")
        logger.info("ros config is:")
        logger.info(ros_config)
        wdir = "." if hiob_path is None else hiob_path
        if environment_path is None:
            self.environment_path = os.path.join(wdir, 'config', 'environment.yaml')
        else:
            self.environment_path = environment_path
        if tracker_path is None:
            self.tracker_path = os.path.join(wdir, 'config', 'tracker.yaml')
        else:
            self.tracker_path = tracker_path

        self.overrides = {}
        if ros_config is not None:
            if (ros_config['subscribe'] is None) != (ros_config['publish'] is None):
                raise Exception("Invalid ros parameters detected! Exiting...")
            else:
                self.overrides['tracking'] = ['ros/' + ros_config['subscribe'].strip('/')]
                self.overrides['ros_node'] = ros_config['publish']
        self.overrides['log_level'] = logging.WARN if silent else logging.INFO
        self.load_files()

    def load_files(self):
        try:
            logger.info(
                "Loading environment file from '%s'", self.environment_path)
            with open(self.environment_path, 'r') as f:
                self.environment = yaml.safe_load(f)
        except IOError as e:
            msg = "Could not open environment configuration file: {} - {}".format(self.environment_path, e)
            logger.error(msg)
            print(msg, file=sys.stderr)
            exit(1)
        try:
            logger.info("Loading tracker file from '%s'", self.tracker_path)
            with open(self.tracker_path, 'r') as f:
                self.tracker = yaml.safe_load(f)
        except IOError as e:
            msg = "Could not open tracker configuration file: {} - {}".format(self.tracker_path, e)
            logger.error(msg)
            print(msg, file=sys.stderr)
            exit(1)

        if 'ros_node' not in self:
            self.set_override("ros_node", None)
        for sname in self['tracking']:
            p1, p2 = sname.split(os.path.sep, 1)
            if p1 == "ros":
                print("Found live sample; enabling ros mode.")
                if self['ros_node'] is None:
                    self.set_override("ros_node", '/hiob/objects/0')

        self.set_override("ros_mode", self['ros_node'] is not None)

        if self['ros_mode'] and len(self['tracking']) > 1:
            print("Ros mode is enabled, but multiple samples have been defined. Disabling all but the first ros sample.")
            self.set_override('tracking', [list(filter(lambda x: len(x) >= 3 and x[:4] == 'ros/', self['tracking']))[0]])

        print("-----------------------------------------------------------------------------")
        print("ros mode is {0}.".format("TRUE" if self["ros_mode"] else "FALSE"))
        if self["ros_mode"]:
            print("subscribing to {}".format(self['tracking'][0]))
            print("publishing to {}".format(self['ros_node']))
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

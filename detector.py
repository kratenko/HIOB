"""
Created on 2017-03-01

@author: Peer Springstübe
"""

# so dirty....

import logging
logger = logging.getLogger(__name__)


def detect(tracking, frame):
    logger.info("Detecting")
    tracker = tracking.tracker

    logger.info("%s, %s, %s", tracker, tracking, frame)
    previous_position = frame.previous_position
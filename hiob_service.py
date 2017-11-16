#!/data/3knoeppl/bsc_thesis/venvs/hiob_new/bin/python
# export MPLBACKEND="agg"
# PYTHONPATH= .. python hy1.py

import logging
import transitions
import argparse
import asyncio
from Configurator import Configurator
from Tracker import Tracker

logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)

def track(environment_path=None, tracker_path=None):

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=environment_path,
        tracker_path=tracker_path
    )

    #create the tracker instance
    logger.info("Creating tracker object")
    tracker = Tracker(conf)
    tracker.setup_environment()

    # create temsorflow session and do the tracking
    logging.info("Initiate tracking process")
    session = tracker.setup_session()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(tracker.execute_everything())

    session.close()
    return tracker.evaluation

def main():
    # parse arguments:
    logger.info("Prasing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment')
    parser.add_argument('t', '--tracker')
    parser.add_argument('E', '--evaluation')
    args = parser.parse_args()

    ev = track(environment_path=args.environment, tracker_path=args.tracker)
    logger.info("Tracking finished!")

import argparse
import logging

import transitions

from Configurator import Configurator
from app import App

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment')
    parser.add_argument('-t', '--tracker')
    args = parser.parse_args()

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=args.environment,
        tracker_path=args.tracker,
    )

    # execute app app and run tracking
    logger.info("Initiate tracking process in app app")
    app = App(logger, conf)
    app.run()

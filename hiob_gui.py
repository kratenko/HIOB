import logging

import transitions
import sys, os

sys.path.append( os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from .core.Configurator import Configurator
from .core.app import App
from .core.argparser import parser

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


def main():
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser.prog = "hiob_gui"
    args = parser.parse_args()

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=args.environment,
        tracker_path=args.tracker,
        ros_config={'subscribe': args.ros_subscribe, 'publish': args.ros_publish}
    )

    # execute app app and run tracking
    logger.info("Initiate tracking process in app app")
    app = App(logger, conf)
    app.run()


if __name__ == '__main__':
    main()

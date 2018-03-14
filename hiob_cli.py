# export MPLBACKEND="agg"
import logging
import transitions
import asyncio
import os, sys


sys.path.append(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from .core.Configurator import Configurator
from .core.Tracker import Tracker
from .core.argparser import parser

# Set up logging:
logging.getLogger().setLevel(logging.INFO)
#logging.getLogger().setLevel(logging.DEBUG)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


def track(environment_path=None, tracker_path=None, ros_config=None):

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=environment_path,
        tracker_path=tracker_path,
        ros_config=ros_config
    )


    # create the tracker instance
    logger.info("Creating tracker object")
    tracker = Tracker(conf)
    tracker.setup_environment()

    # create tensorflow session and do the tracking
    logger.info("Initiate tracking process")
    loop = asyncio.get_event_loop()
    with tracker.setup_session():
        loop.run_until_complete(tracker.execute_everything())
    loop.close()

    # return the evaluation results (an OrderedDict)
    return tracker.evaluation


def main():
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser.prog = "hiob_cli"
    args = parser.parse_args()

    ev = track(environment_path=args.environment, tracker_path=args.tracker,
               ros_config={'subscribe': args.ros_subscribe, 'publish': args.ros_publish})
    logger.info("Tracking finished!")
    ev_lines = "\n  - ".join(["{}={}".format(k, v) for k, v in ev.items()])
    logger.info("Evaluation:\n  - %s", ev_lines)
    # copy evaluation to file
    if args.evaluation is not None:
        path = args.evaluation
        logger.info("Copying evaluation to '%s'", path)
        with open(path, "w") as f:
            f.write(ev_lines + "\n")


if __name__ == '__main__':
    main()

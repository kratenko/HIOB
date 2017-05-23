#!/usr/bin/env/python
# export MPLBACKEND="agg"
# PYTHONPATH=.. python hy1.py
import logging
import transitions

# Set up logging:
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


def track(environment_path=None, tracker_path=None):
    from hiob.configuration import Configurator
    from hiob.tracker import Tracker

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=environment_path,
        tracker_path=tracker_path,
    )

    # create the tracker instance
    logger.info("Creating tracker object")
    tracker = Tracker(conf)
    tracker.setup_environment()

    # create tensorflow session and do the tracking
    logger.info("Initiate tracking process")
    with tracker.setup_session():
        tracker.execute_everything()

    # return the evaluation results (an OrderedDict)
    return tracker.evaluation


def main():
    import argparse

    # parse arguments:
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment')
    parser.add_argument('-t', '--tracker')
    parser.add_argument('-E', '--evaluation')
    args = parser.parse_args()

    ev = track(environment_path=args.environment, tracker_path=args.tracker)
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

import rospy
import logging
import hiob_msgs.msg


logger = logging.getLogger(__name__)


class RosPositionPublisher:

    def __init__(self, node_name):
        self._started = False
        logger.info('-- Init Hiob:RosPositionPublisher --')
        logger.debug('init ROS publisher')
        self._publisher = rospy.Publisher(node_name, hiob_msgs.msg.TrackingResult, queue_size=1)
        rospy.on_shutdown(self.stop)
        logger.info('-- Done --')

    def start(self):
        if self._started:
            logger.warning('Publisher already running')
            return
        self._started = True

    def stop(self):
        self._started = False

    def is_running(self):
        return self._started

    def publish(self, tracking_result):
        logger.info("publishing tracking result...")
        if not self._started:
            self.start()
        #    raise PublisherTerminatedError
        logger.info(str(tracking_result))
        pos = tracking_result['predicted_position']
        pos_msg = hiob_msgs.msg.Rect(pos.x, pos.y, pos.w, pos.h)
        frame_id = tracking_result['frame_id'] if 'frame_id' in tracking_result else None
        result_msg = hiob_msgs.msg.TrackingResult(
            pos_msg,
            tracking_result['prediction_quality'],
            tracking_result['lost'],
            frame_id)
        self._publisher.publish(result_msg)


class PublisherTerminatedError(Exception):

    def __init__(self):
        super().__init__(
            "RosPositionPublisher.publish() was called, but the publisher has been terminated!")

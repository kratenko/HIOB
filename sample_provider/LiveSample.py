import io
from PIL import Image
import rospy
import sensor_msgs.msg
import threading
#import cv2
#from cv_bridge import CvBridge
from hiob.Rect import Rect


class LiveSample:

    def __init__(self, node_id):
        self.node_id = node_id
        self._buffer = []
        self.images = []
        self.current_frame_id = 0
        self.frames_skipped = 0
        self.subscriber = None
        self.loaded = True
        self.full_name = 'ros/' + node_id
        self.ros_event = threading.Event()
        self.initial_position = Rect(0, 0, 50, 50)
        self.set_name = 'ros'
        self.name = self.node_id
        #self._bridge = CvBridge()
        rospy.on_shutdown(self.unload)

    def __repr__(self):
        return '<ROS::{node}>'.format(node=self.node_id)

    def load(self):
        rospy.init_node("hiob_subscriber", anonymous=True)
        self.subscriber = rospy.Subscriber(self.node_id, sensor_msgs.msg.CompressedImage, self.receive_frame)

    def unload(self):
        if self.loaded:
            self.ros_event.set()
            if self.subscriber:
                self.subscriber.unregister()
            self.images = []
            self.loaded = False

    def receive_frame(self, msg):
        print("received frame!")
        #cv_img = self._bridge.imgmsg_to_cv2(msg)
        #img = Image.open(io.BytesIO(bytearray(msg)))
        #img = Image.fromarray(cv_img)
        img = Image.open(io.BytesIO(bytearray(msg.data)))
        if img.mode != "RGB":
            # convert s/w to colour:
            img = img.convert("RGB")
        img.show()
        self._buffer.append(img)
        self.ros_event.set()

    async def get_next_frame_data(self):
        self.current_frame_id = len(self.images)
        while len(self._buffer) == 0 and self.loaded:
            self.ros_event.wait()
            print("callback fired!")
            self.ros_event.clear()
        else:
            self.frames_skipped += len(self._buffer) - 1
            self.images.append(self._buffer[-1])
            self._buffer = []

        return [
            self.images[-1],
            None]

    def frames_left(self):
        return 1 if self.loaded else 0

    def count_frames_processed(self):
        return len(self.images)

    def count_frames_skipped(self):
        return self.frames_skipped

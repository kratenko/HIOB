import io
from PIL import Image
import rospy
import sensor_msgs.msg
import threading
import tempfile
import os.path
import subprocess
from hiob.Rect import Rect
import nico_dummy_vision.msg


class LiveSample:

    def __init__(self, node_id):
        self.node_id = node_id
        self._buffer = []
        self.images = []
        self.attributes = []
        self.ground_truth = []
        self.current_frame_id = 0
        self.frames_skipped = 0
        self.subscriber = None
        self.loaded = True
        self.full_name = 'ros/' + node_id
        self.ros_event = threading.Event()
        self.initial_position = None
        self.set_name = '__ros__'
        self.name = self.node_id
        self._img_path = os.path.join(tempfile.gettempdir(), 'hiob.received.png')
        #self._bridge = CvBridge()
        rospy.on_shutdown(self.unload)

    def __repr__(self):
        return '<ROS::{node}>'.format(node=self.node_id)

    def load(self):
        self.subscriber = rospy.Subscriber(self.node_id, nico_dummy_vision.msg.FrameWithGroundTruth, self.receive_frame)
        self.images = []

    def unload(self):
        if self.loaded:
            self.ros_event.set()
            if self.subscriber:
                self.subscriber.unregister()
            #self.images = []
            self.loaded = False

    def receive_frame(self, msg):
        print("received frame!")
        #cv_img = self._bridge.imgmsg_to_cv2(msg)
        #img = Image.open(io.BytesIO(bytearray(msg)))
        #img = Image.fromarray(cv_img)
        if msg.lastImage:
            self.unload()
        img = Image.open(io.BytesIO(bytearray(msg.frame.data)))
        if img.mode != "RGB":
            # convert s/w to colour:
            img = img.convert("RGB")

        #img.show()
        gt = msg.groundTruth
        if gt:
            rect = Rect(gt.x, gt.y, gt.w, gt.h)
            self._buffer.append((img, rect))
            if self.initial_position is None:
                print("Updating initial position... (" + str(rect) + "/" + str(gt) + ")")
                self.initial_position = rect
        else:
            self._buffer.append((img, None))
        print("buffer length: " + str(len(self._buffer)))
        self.ros_event.set()

    async def get_next_frame_data(self):
        self.current_frame_id = len(self.images)
        while len(self._buffer) == 0 and self.loaded:
            self.ros_event.wait()
            print("callback fired!")
            self.ros_event.clear()
        else:
            #if not self.loaded:
            #    raise BaseException("not loaded!")
            self.frames_skipped += len(self._buffer) - 1
            self.images.append(self._buffer[-1][0])
            self.ground_truth.append(self._buffer[-1][1])
            self._buffer = []
        # if self.initial_position is None and self.ground_truth[-1] is not None:
            # self.initial_position = self.ground_truth[-1]

        return [
            self.images[-1],
            self.ground_truth[-1]]

    def frames_left(self):
        return 1 if self.loaded else 0

    def count_frames_processed(self):
        return len(self.images)

    def count_frames_skipped(self):
        return self.frames_skipped

    def get_actual_frames(self):
        return len(self.images)

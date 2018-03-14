import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--environment')
parser.add_argument('-t', '--tracker')
parser.add_argument('-E', '--evaluation')
parser.add_argument('--ros-subscribe', default=None, type=str, dest='ros_subscribe')
parser.add_argument('--fake-fps', default=0, type=int, dest='fake_fps')

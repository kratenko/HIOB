"""
Created on 2016-09-01

@author: Peer Springstübe

python -u fcnt_matlab.py 2>&1 | tee thelongrun.log
"""

MATLAB_BIN = "/opt/MATLAB/R2015a/bin/matlab"
FCNT_PATH = "/data/springstuebe/FCNT-fork"
DATA_PATH = "/data/springstuebe/data/"
UNZIPPED_PATH = DATA_PATH + "tb100_unzipped/"
import os
import subprocess
import datetime


liste = """Basketball
Biker
Bird1
Bird2
BlurBody
BlurCar1
BlurCar2
BlurCar3
BlurCar4
BlurFace
BlurOwl
Board
Bolt
Bolt2
Box
Boy
Car1
Car2
Car24
Car4
CarDark
CarScale
ClifBar
Coke
Couple
Coupon
Crossing
Crowds
Dancer
Dancer2
David
David2
David3
Deer
Diving
Dog
Dog1
Doll
DragonBaby
Dudek
FaceOcc1
FaceOcc2
Fish
FleetFace
Football
Football1
Freeman1
Freeman3
Freeman4
Girl
Girl2
Gym
Human2
Human3
Human4.2
Human5
Human6
Human7
Human8
Human9
Ironman
Jogging.1
Jogging.2
Jump
Jumping
KiteSurf
Lemming
Liquor
Man
Matrix
Mhyang
MotorRolling
MountainBike
Panda
RedTeam
Rubik
Shaking
Singer1
Singer2
Skater
Skater2
Skating1
Skating2.1
Skating2.2
Skiing
Soccer
Subway
Surfer
Suv
Sylvester
Tiger1
Tiger2
Toy
Trans
Trellis
Twinnings
Vase
Walking
Walking2
Woman"""


VERSION = "2"

def runf(sample_name):
    cmd = "cd {} && {} -nodisplay -nodesktop -r \"data_path=['{}']; seq_name='{}'; runf; exit;\"".format(
        FCNT_PATH, MATLAB_BIN, UNZIPPED_PATH, sample_name)
    begin = datetime.datetime.now()
    print(begin, "SAMPLE BEGIN: '%s'" % sample_name)
    print(cmd)
    subprocess.call(cmd, shell=True)
    end = datetime.datetime.now()
    print(end, "SAMPLE DONE: '%s', duration: '%s'" %
          (sample_name, end - begin))

#samples = sorted(os.listdir(UNZIPPED_PATH))
#samples = ["Deer", "MotorRolling"]
samples = liste.split("\n")
print("FCNT caller Version %s" % VERSION)
print(samples)

for s in samples:
    print(s)
    data_path = UNZIPPED_PATH
    seq_name = s
    runf(s)

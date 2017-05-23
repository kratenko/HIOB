"""
Created on 2016-12-08

@author: Peer Springstübe
"""

file1 = "/data/Peer/data/tb100-attributes.txt"

samples = {}


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


samples = {n.strip(): [] for n in liste.split('\n')}


with open(file1, "r") as f:
    for line in f.readlines():
        if ':' not in line:
            continue
        att, name_list = line.split(':')
        att = att.strip()
        names = name_list.split(',')
        for name in names:
            name = name.strip()
            samples[name].append(att)
print("name: tb100")
print("samples:")
for name in sorted(samples.keys()):
    print("    - name: %s" % name)
    print("      attributes: [%s]" % ', '.join(sorted(samples[name])))

from itertools import cycle
try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk

from PIL import ImageTk, ImageDraw, ImageFont

from vis.DataSample import DataSample
import os

# DATA_PATH = '/informatik2/students/home/3springs/Diplom/data'
#DATA_PATH = '/media/3springs/USB DISK/Diplom/tb50'
DATA_PATH = '/data/Peer/data/'
PRINCETON_PATH = os.path.join(os.sep, 'data', 'Peer', 'data', 'princeton', )


class App(tk.Tk):
    '''Tk window/label adjusts to size of image'''

    def __init__(self, ds, x, y, delay):
        # the root will be self
        tk.Tk.__init__(self)
        # set x, y position only
        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay
        # allows repeat cycling through the pictures
        # store as (img_object, img_name) tuple
        # self.pictures = cycle(ImageTk.PhotoImage(frame)
        #                      for frame in ds.frames)
        #self.ds = ds
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/freefont/FreeMonoBold.otf", 64)
        #self.font = ImageFont.load_default()
        self._prep(ds)
        #self.frames = cycle(ds.frames)
        self.pictures = cycle(ImageTk.PhotoImage(frame)
                              for frame in self.frames)
        self.picture_display = tk.Label(self)
        self.picture_display.pack()

    def _prep(self, ds):
        frames = []
        for n, fr in enumerate(ds.frames):
            draw = ImageDraw.Draw(fr)
            # draw frame number:
            draw.text((0, 0), "#%d/%d" %
                      (n, len(ds.frames)), "white", font=self.font)
            # draw ground truth(s):
            for ngt, gtt in enumerate(ds.groundtruths):
                if n >= len(gtt):
                    continue
                print(n, len(gtt))
                gt = gtt[n]
                # draw rect from gt
                if gt:
                    print(gt)
                    r = (gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])
                    print(n, gt, r)
                    draw.rectangle(r, None, (255 // (ngt + 1), 255, 255, 255))
            frames.append(fr)
            # fcnt tracking result:
            if ds.fcnt_position:
                p = ds.fcnt_position[n]
                r = [int(p[0]), int(p[1]), int(p[2] + p[0]), int(p[3] + p[1])]
                draw.rectangle(r, None, (255 // (ngt + 1), 255, 0, 255))

        self.frames = frames

    def show_slides(self):
        '''cycle through the images and show them'''
        # next works with Python26 or higher
        #frame = next(self.frames)
        #frame = self.ds.frames[0]
        # print frame
        #img_object = ImageTk.PhotoImage(frame)
        #img_object = next(self.pictures)
        img_object = next(self.pictures)
        # print img_object
        self.picture_display.config(image=img_object)
        # shows the image filename, but could be expanded
        # to show an associated description of the image
        self.after(self.delay, self.show_slides)

    def run(self):
        self.mainloop()


def load_sample(name):
    zip_path = DATA_PATH + name
    print("loading", zip_path)
    #zip_path = DATA_PATH + '/.zip'
    ds = DataSample()
    ds.load_from_zip(zip_path)
    # print ds.groundtruth_paths
    # print ds.groundtruths
    return ds


def load_dir_sample(name):
    dir_path = DATA_PATH + name
    print("loading", dir_path)
    ds = DataSample()
    ds.load_from_dir(dir_path)
    # print ds.groundtruth_paths
    # print ds.groundtruths
    return ds


def load_princeton_sample(data_set, name):
    print("loading", name)
    if data_set == 'evaluation':
        path = os.path.join(PRINCETON_PATH, 'EvaluationSet', name)
    elif data_set == 'validation':
        path = os.path.join(PRINCETON_PATH, 'ValidationSet', name)
    ds = DataSample()
    ds.load_princeton(path)
    print(ds.groundtruths)
    return ds


def main(name, data_set='data'):
    # set milliseconds time between slides
    delay = int(1000.0 / 30.0)
    # get a series of gif images you have in the working folder
    # or use full path, or set directory to where the images are
    # upper left corner coordinates of app window
    x = 100
    y = 50
    if data_set in ['evaluation', 'validation']:
        # princeton evaluation set
        ds = load_princeton_sample(data_set, name)
    elif data_set == 'dir':
        ds = load_dir_sample(name)
    else:
        ds = load_sample(name)
    app = App(ds, x, y, delay)
    app.show_slides()
    app.run()

evaluation = """bag1
basketball1
basketball2
basketball2.2
basketballnew
bdog_occ2
bear_back
bear_change
bird1.1_no
bird3.1_no
book_move1
book_turn
book_turn2
box_no_occ
br_occ_0
br_occ1
br_occ_turn0
cafe_occ1
cc_occ1
cf_difficult
cf_no_occ
cf_occ2
cf_occ3
child_no2
computerbar1
computerBar2
cup_book
dog_no_1
dog_occ_2
dog_occ_3
express1_occ
express2_occ
express3_static_occ
face_move1
face_occ2
face_occ3
face_turn2
flower_red_occ
gre_book
hand_no_occ
hand_occ
library2.1_occ
library2.2_occ
mouse_no1
new_ex_no_occ
new_ex_occ1
new_ex_occ2
new_ex_occ3
new_ex_occ5_long
new_ex_occ6
new_ex_occ7.1
new_student_center1
new_student_center2
new_student_center3
new_student_center4
new_student_center_no_occ
new_ye_no_occ
new_ye_occ
one_book_move
rose1.2
static_sign1
studentcenter2.1
studentcenter3.1
studentcenter3.2
three_people
toy_car_no
toy_car_occ
toy_green_occ
toy_mo_occ
toy_no
toy_no_occ
toy_wg_no_occ
toy_wg_occ
toy_wg_occ1
toy_yellow_no
tracking4
tracking7.1
two_book
two_dog_occ1
two_people_1.1
two_people_1.2
two_people_1.3
walking_no_occ
walking_occ1
walking_occ_long
wdog_no1
wdog_occ3
wr_no
wr_no1
wr_occ2
wuguiTwo_no
zball_no1
zball_no2
zball_no3
zballpat_no1"""

evaluation_list = evaluation.split("\n")

validation = """bear_front
child_no1
face_occ5
new_ex_occ4
zcup_move_1"""


validation_list = validation.split("\n")

# main("tb50/Skiing.zip")
# exit()
# for n in validation_list:
#    main(n, 'validation')
# exit()

# main("tb100/Jogging.zip")

tb50s = """tb50/Basketball.zip
tb50/Biker.zip
tb50/Bird1.zip
tb50/BlurBody.zip
tb50/BlurCar2.zip
tb50/BlurFace.zip
tb50/BlurOwl.zip
tb50/Bolt.zip
tb50/Box.zip
tb50/Car1.zip
tb50/Car4.zip
tb50/CarDark.zip
tb50/CarScale.zip
tb50/ClifBar.zip
tb50/Couple.zip
tb50/Crowds.zip
tb50/David.zip
tb50/Deer.zip
tb50/Diving.zip
tb50/DragonBaby.zip
tb50/Dudek.zip
tb50/Football.zip
tb50/Freeman4.zip
tb50/Girl.zip
tb50/Human3.zip
tb50/Human4.zip
tb50/Human6.zip
tb50/Human9.zip
tb50/Ironman.zip
tb50/Jumping.zip
tb50/Jump.zip
tb50/Liquor.zip
tb50/Matrix.zip
tb50/MotorRolling.zip
tb50/Panda.zip
tb50/RedTeam.zip
tb50/Shaking.zip
tb50/Singer2.zip
tb50/Skating1.zip
tb50/Skating2.zip
tb50/Skiing.zip
tb50/Soccer.zip
tb50/Surfer.zip
tb50/Sylvester.zip
tb50/Tiger2.zip
tb50/Trellis.zip
tb50/Walking2.zip
tb50/Walking.zip
tb50/Woman.zip"""

tb50l = tb50s.split("\n")
print(tb50l)

tb100s = """tb100/Bird2.zip
tb100/BlurCar1.zip
tb100/BlurCar3.zip
tb100/BlurCar4.zip
tb100/Board.zip
tb100/Bolt2.zip
tb100/Boy.zip
tb100/Car24.zip
tb100/Car2.zip
tb100/Coke.zip
tb100/Coupon.zip
tb100/Crossing.zip
tb100/Dancer2.zip
tb100/Dancer.zip
tb100/David2.zip
tb100/David3.zip
tb100/Dog1.zip
tb100/Dog.zip
tb100/Doll.zip
tb100/FaceOcc1.zip
tb100/FaceOcc2.zip
tb100/Fish.zip
tb100/FleetFace.zip
tb100/Football1.zip
tb100/Freeman1.zip
tb100/Freeman3.zip
tb100/Girl2.zip
tb100/Gym.zip
tb100/Human2.zip
tb100/Human5.zip
tb100/Human7.zip
tb100/Human8.zip
tb100/Jogging.zip
tb100/KiteSurf.zip
tb100/Lemming.zip
tb100/Man.zip
tb100/Mhyang.zip
tb100/MountainBike.zip
tb100/Rubik.zip
tb100/Singer1.zip
tb100/Skater2.zip
tb100/Skater.zip
tb100/Subway.zip
tb100/Suv.zip
tb100/Tiger1.zip
tb100/Toy.zip
tb100/Trans.zip
tb100/Twinnings.zip
tb100/Vase.zip"""

tb100s = """tb100/Rubik.zip
tb100/Singer1.zip
tb100/Skater2.zip
tb100/Skater.zip
tb100/Subway.zip
tb100/Suv.zip
tb100/Tiger1.zip
tb100/Toy.zip
tb100/Trans.zip
tb100/Twinnings.zip
tb100/Vase.zip"""


# main("tb100/Singer1.zip")
main("tb100_unzipped/MotorRolling", data_set='dir')
exit()


tb100l = tb100s.split("\n")
print(tb100l)

for n in tb100l:
    main(n)

for n in validation_list:
    main(n, 'validation')

import logging
from hiob.vgg16 import Vgg16
from hiob.sample import Sample

logging.basicConfig(level=logging.DEBUG)


img = "/home/kratenko/git/x_hiob/x_hiob/human13.jpeg"
img_path = "/home/kratenko/cam/sc"

img = img_path + '/' + "DSC07121.JPG"

img = '/data/Peer/ILSVRC2012_val_00040013.JPEG'


s = Sample()
s.load_original_image(img)
s.representations['original'].image.resize((224, 224)).show()

vgg16 = Vgg16()
vgg16.load_model()

cl = vgg16.predict_sample(s)

print("Best guesses:")
for n, guess in enumerate(cl.best_guesses(10)):
    print("%02d: (%03d/%6.4f) %s" %
          (n + 1, guess[1], guess[0], guess[2]))

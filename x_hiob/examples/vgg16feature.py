import logging
from x_hiob.vgg16 import Vgg16F
from x_hiob.sample import Sample
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


image_path = '/data/Peer/ILSVRC2012_val_00040013.JPEG'
image_path = '/data/Peer/Jogging/img/0027.jpg'


s = Sample()
s.load_original_image(image_path)
s.representations['original'].image.resize((368, 368)).show()

net = Vgg16F(input_size=(368, 368))
net.load_model()

fm = net.predict_sample(s)

s.representations['Vgg16F/conv4_3'].plot(2)
plt.show()

#print("Best guesses:")
# for n, guess in enumerate(cl.best_guesses(10)):
#    print("%02d: (%03d/%6.4f) %s" %
#          (n + 1, guess[1], guess[0], guess[2]))

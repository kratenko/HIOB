# -*- cos utf-8 -*-
"""
Created on Fri Jul 22 14:11:29 2016

@author: flo
"""

from __future__ import print_function # so that print(a, b) isnt interpreted as printing a tuple in Python 2
from __future__ import division # so that 5 / 2 == 2.5 in both Python 2 and 3


import glob
import numpy as np
import tensorflow as tf
#import cv2
import scipy.misc
import scipy.ndimage

from future.moves.itertools import zip_longest # fix Python 2 and 3 naming differences

def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def tinymodel(X, weights_conv, bias_conv, weights_fc, bias_fc, weights_conv2, bias_conv2):
    conv = tf.nn.conv3d(X, weights_conv, strides=[1, 1, 1, 1, 1], padding='SAME') + bias_conv
    conv = tf.Print(conv, [conv], message="C1: ")

    activations = tf.nn.relu(conv)
    activations = tf.Print(activations, [activations], message="A1: ")

    pool = tf.nn.max_pool3d(activations, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                            padding='SAME')  # -> 60, 80, 8, 4
    print("pool1 shape:", pool.get_shape())

    conv2 = tf.nn.conv3d(pool, weights_conv2, strides=[1, 1, 1, 1, 1], padding='SAME') + bias_conv2
    activations2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool3d(activations2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    pool2 = tf.Print(pool2, [pool2], message="P2: ")
    print("pool2 shape:", pool2.get_shape())

    pool2_flat = tf.reshape(pool2, [-1, weights_fc.get_shape().as_list()[0]])  # flatten to fit fc layer
    mul = tf.matmul(pool2_flat, weights_fc)

    print("mul shape:", mul.get_shape())

    return mul + bias_fc  # is this correct? no idea :(


# def model(X, weights_conv, bias_conv, weights_conv2, bias_conv2, weights_conv3, bias_conv3, weights_fc, bias_fc):
#
#     # INPUT X : image sequence
#
#     # CONV layer will compute the output of neurons that are connected to local regions in the input
#     conv = tf.nn.conv3d(X, weights_conv, strides=[1, 1, 1, 1, 1], padding='SAME') + bias_conv
#
#     conv = tf.Print(conv, [conv], message="C1: ")
#
#     # RELU layer will apply an elementwise activation function, such as the max(0,x)max(0,x) thresholding at zero.
#     # This leaves the size of the volume unchanged ([32x32x32]).
#     activations = tf.nn.relu(conv)  # activations shape=(32, 32, 32)
#     #print("conv&relu shape:", activations.get_shape())
#     activations = tf.Print(activations, [activations], message="A1: ")
#
#     # POOL layer will perform a downsampling operation along the spatial dimensions (width, height),
#     # resulting in volume such as [16x16x32].
#     pool = tf.nn.max_pool3d(activations, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME') # -> 16, 16, 32
#     #print("pool shape:", pool.get_shape())
#     pool = tf.Print(pool, [pool], message="P1: ")
#
#     conv2 = tf.nn.conv3d(pool, weights_conv2, strides=[1, 1, 1, 1, 1], padding='SAME') + bias_conv2
#     activations2 = tf.nn.relu(conv2)
#     pool2 = tf.nn.max_pool3d(activations2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
#     pool2 = tf.Print(pool2, [pool2], message="P2: ")
#
#     conv3 = tf.nn.conv3d(pool2, weights_conv3, strides=[1, 1, 1, 1, 1], padding='SAME') + bias_conv3
#     activations3 = tf.nn.relu(conv3)
#     pool3 = tf.nn.max_pool3d(activations3, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
#     pool3 = tf.Print(pool3, [pool3], message="P3: ")
#
#     # TODO: second fully connected layer with dropout?
#     # could add additional fully connected layer + dropout here (e.g with 1024 neurons, like in MNIST tutorial:
#     # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html)
#     # pool_flat = tf.reshape(pool, [-1, 16*16*32]) # flatten result of pooling
#     # activations_fc0 = tf.nn.relu(tf.matmul(pool_flat, weights_fc) + bias_fc)
#
#     # FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10],
#     # where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10.
#     pool3_flat = tf.reshape(pool3, [-1, weights_fc.get_shape().as_list()[0]]) # flatten to fit fc layer
#     mul = tf.matmul(pool3_flat, weights_fc)
#
#     #return tf.nn.softmax(mul + bias_fc) -> NO! perform softmax when computing cross entropy instead (see below)
#
#     return mul + bias_fc # is this correct? no idea :(

def get_batch(config, training=True):
    jobs = config['jobs']
    split = 0.8

    folders = glob.glob(jobs + "/*") # FIXME: shuffle predictably!
    split_n = int(len(folders) * split)

    if training:
        return folders[0:split_n]
    else:
        return folders[split_n:]


def get_teacher_from_folder(sess, folder, pl_filename, pl_img):
    return sess.run(pl_img, feed_dict={pl_filename: "%s/_teacher.png" % folder})


def get_sequence_from_folder(folder, fraction=1.0, fraction_teacher=1.0):
    """
    Read image sequence and teacher signal from a given folder.

    :param folder: Path to folder
    :param fraction: The fraction to downscale images (1=original size, 0.5=each dimension half size)
    :return: tuple of sequence and teacher image:
        (sequence, teacher)
        | sequence is a numpy array of images
        | teacher is a numpy array
    """
    files = [folder + "/depth-%d.png" % i for i in range(15)]
    file_teacher = folder + '/_teacher.png'

    # sequence = np.array([scipy.misc.imresize(scipy.ndimage.imread(fn, mode='I'), fraction) for fn in files])
    sequence = np.array([scipy.ndimage.imread(fn, mode='I') for fn in files])
    sequence = np.swapaxes(np.swapaxes(sequence, 0, 1), 2, 1) # reorder so that (480, 640, 15) -> (15, 480, 640)

    # teacher = scipy.misc.imresize(scipy.ndimage.imread(file_teacher, mode='I'), fraction*fraction_teacher)
    teacher = scipy.ndimage.imread(file_teacher, mode='I')

    return sequence, teacher

def get_tensor_from_folder(sess, folder, pl_filename, pl_img, pl_teacher_filename, pl_teacher):
    files = [folder + "/depth-%d.png" % i for i in range(15)]
    file_teacher = folder + '/_teacher.png'
    print(file_teacher)
    teacher = sess.run(pl_teacher, feed_dict={pl_teacher_filename: file_teacher})

    images_array = np.zeros((480, 640, 15), np.float32)
    for i, image in enumerate(files):
        images_array[:, :, i] = sess.run(pl_img, feed_dict={pl_filename: image})
    return images_array, teacher

# def get_batch_from_folders(sess, folders, pl_batch, pl_filename, pl_img):
#     batch_size = len(folders)
#     batch = np.zeros((batch_size, 480, 640, 15, 1), np.float32)
#     for i, folder in enumerate(folders):
#         tensor = get_tensor_from_folder(sess, folder, pl_filename, pl_img)
#         batch[i, :, :, :, 0] = tensor
#     return batch

# def grouper(the_list, size):
#    return zip(*(iter(the_list),) * size)


def grouper(iterable, size, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * size
    return zip_longest(fillvalue=fillvalue, *args)

def read_foldernames(folders, batch_size, fraction, num_frames, channels, fraction_teacher):
    sequences = np.zeros(shape=(batch_size, 480 * fraction, 640 * fraction, num_frames, channels), dtype=np.float32)
    teachers = np.zeros(shape=(batch_size, 480 * fraction * fraction_teacher, 640 * fraction * fraction_teacher),
                        dtype=np.float32)

    for i, folder in enumerate(folders[:batch_size]):
        sequence, teacher = get_sequence_from_folder(folder, fraction=fraction, fraction_teacher=fraction_teacher)

        sequences[i, :, :, :, 0] = sequence
        teachers[i, :, :] = teacher

    return sequences, teachers


def main(config):

    # model parameters
    fraction = 0.25  # downscale images
    fraction_teacher = 0.1 # further shrink down output layer (on top of fraction shrinking)
    num_features = 4
    batch_size = 10
    test_size = batch_size
    num_frames = 15  # frames per gesture window
    channels = 1  # only depth channel

    weights_conv = weight_variable([3, 3, num_frames, channels, num_features], name="weights-conv") # TODO: num_frames -> 3
    bias_conv = bias_variable([num_features], name="bias-conv")  # bias variable for each feature=output channel

    weights_conv2 = weight_variable([3, 3, 3, num_features, num_features * 2], name="weights-conv-2")
    bias_conv2 = bias_variable([num_features * 2], name="bias-conv-2")

    weights_conv3 = weight_variable([3, 3, 3, num_features * 2, num_features * 2], name="weights-conv-3")
    bias_conv3 = bias_variable([num_features * 2], name="bias-conv-3")

    # for one layer: 60*80*8*4
    # 4: result of conv+pool over time domain... TODO: why 4?
    weights_fc = weight_variable([30*40*4*num_features*2, int(640*fraction*fraction_teacher) * int(480*fraction*fraction_teacher)],
                                 name="weights-fc")  # output of pooling -> fully connected layer
    bias_fc = bias_variable([int(640*fraction*fraction_teacher) * int(480*fraction*fraction_teacher)], name="bias-fc")  # bias for every output neuron # FIXME: what size is this? 2d maybe?

    # things needed to read in image sequences

    # filename = tf.placeholder("string")
    # png_string = tf.read_file(filename)
    # img = tf.image.decode_png(png_string)
    # img_float = tf.cast(img, tf.float32)
    # img_float = tf.reshape(img_float, (480, 640))
    #
    # teacher_filename = tf.placeholder("string")
    # png_teacher = tf.read_file(teacher_filename)
    # teacher_img = tf.image.decode_png(png_teacher)
    # teacher_float = tf.cast(teacher_img, tf.float32)
    # teacher_float = tf.reshape(teacher_float, (480, 640))

    images = tf.placeholder(tf.float32, shape=[None, int(480*fraction), int(640*fraction), num_frames, channels]) # None: unknown batch size. 1: one input channel
    pl_teachers = tf.placeholder(tf.float32, shape=[None, int(480*fraction*fraction_teacher), int(640*fraction*fraction_teacher)])
    # output = model(images, weights_conv, bias_conv, weights_conv2, bias_conv2, weights_conv3, bias_conv3, weights_fc, bias_fc)
    tinyoutput = tinymodel(images, weights_conv, bias_conv, weights_fc, bias_fc, weights_conv2, bias_conv2)

    # DEFINE LEARNING

    teachers_flat = tf.reshape(pl_teachers, [batch_size, int(480*fraction*fraction_teacher) * int(640*fraction*fraction_teacher)]) # fit fc layer FIXME: why doesnt this work? [-1, int(480*fraction) * int(640*fraction)]
    # teachers_flat = pl_teachers
    print("SHAPES", tinyoutput.get_shape(), teachers_flat.get_shape())
    cost = tf.reduce_mean(tf.square(tinyoutput - teachers_flat))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    tinytrain = optimizer.minimize(cost)

    with tf.Session() as sess:
        print("[busy] starting queue runners...")
        tf.train.start_queue_runners()
        print("[done]")

        print("[busy] initializing all variables...")
        # init = tf.initialize_all_variables()
        init = tf.initialize_variables([weights_conv, bias_conv, weights_fc, bias_fc, weights_conv2, bias_conv2])
        sess.run(init)
        print("[done] variables initialized...")

        print("[busy] Preparing data...")
        folders = get_batch(config, True)
        print("[done] Data prepared")

        folder_groups = grouper(folders, batch_size)
        for i, batch in enumerate(folder_groups):
            print("[busy] Reading batch #%d of %d" % (i, len(folders)/batch_size))
            print("[info] Jobs: %s" % ('\n - '.join([b.split('/')[-1] for b in batch])))
            sequences, teachers = read_foldernames(batch, batch_size, fraction, num_frames, channels, fraction_teacher)

            # print("[stat] shapes (pl_sequence / pl_teacher): ", images.get_shape(), pl_teachers.get_shape())
            # print("Size: %d" % sum([seq.nbytes for seq in sequences]))
            #
            # print("[stat] Shapes. sequences / teachers: ", sequences.shape, teachers.shape)
            # print("[stat] Memory usage. sequences: %d bytes. teachers: %d bytes" % (sequences.nbytes, teachers.nbytes))

            # out = sess.run(train, feed_dict={images: np.array(sequences), teachers: np.array(teachers)})
            print("[busy] Training")
            out = sess.run(tinytrain, feed_dict={images: sequences, pl_teachers: teachers})
            print("[done]")

            # FINAL OUTPUT

            print("[busy] Calculating loss...")
            test_folders = get_batch(config, False)
            test_sequences, test_teachers = read_foldernames(test_folders[:test_size], batch_size, fraction, num_frames, channels, fraction_teacher)
            loss = sess.run(cost, feed_dict={images: test_sequences, pl_teachers: test_teachers})
            print("[done] LOSS:")
            print(loss)

        # out = sess.run(output, feed_dict={images: np.array(sequences), teachers: np.array(teachers)})

        # for i, folder_group in enumerate(grouper(folders, batch_size)):
        #
        #     batch = get_batch_from_folders(sess, folder_group, x, filename, img_float)
        #     out = sess.run(output, feed_dict={images: batch})
        #
        #
        #     # sequence = get_tensor_from_folder(sess, folder, filename, img_float)
        #
        #     if i % 10 == 0:
        #         print("%d of %d | OUT SHAPE:" % (i, len(folders)), out.shape)

if __name__ == '__main__':

    import json

    try:
        with open('config.json') as fp:
            config = json.load(fp)
    except:
        config = {'jobs' : "/Users/flo/projects/master-thesis/code/jobs"}

    print(config)

    main(config)

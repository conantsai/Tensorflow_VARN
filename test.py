import tensorflow as tf
import numpy as np
import os
import random
import PIL.Image as Image
import cv2
import os
import copy
import time
import math
import P3D
import DataGenerator
from settings import *

#Cell for Testing
MOVING_AVERAGE_DECAY=0.99
tf.reset_default_graph()
#when testing ,make sure IS_TRAIN==False,or you will get bad result for testing.
IS_TRAIN = False
final_acc = 0
IS_DA = False
testloader = DataGenerator.DataGenerator(filename='test.list',
                                         batch_size=BATCH_SIZE,
                                         num_frames_per_clip=NUM_FRAMES_PER_CLIP,
                                         shuffle=False,is_da=IS_DA)

c = 0

def compute_accuracy(logit, labels):
    correct = tf.equal(tf.argmax(logit, 1), labels)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc

with tf.Graph().as_default():   
    global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
    video_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_FRAMES_PER_CLIP, CROP_SIZE,CROP_SIZE, RGB_CHANNEL))
    audio_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, AUDIO_SIZE, AUDIO_SIZE))
    label_placeholder = tf.placeholder(tf.int64,shape=(BATCH_SIZE))
    score_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE))

    # when testing,make sure dropout=1.0(keep_prob)
    logit1, logit2 = P3D.inference_varn(video_placeholder, audio_placeholder, 1, BATCH_SIZE)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    acc = compute_accuracy(logit1 ,label_placeholder)
    init = tf.global_variables_initializer()
    variable_avg_restore = variable_averages.variables_to_restore()
    
    avglist = []

    saver = tf.train.Saver(tf.global_variables())
    # You can also restore the moving_average parameters ,like this:
    # saver=tf.train.Saver(variable_avg_restore)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    #restore your checkpoint file
    model_file = tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess, model_file)

    for step in range(math.ceil(testloader.len/BATCH_SIZE)):
        image, audio, label, score, _ = testloader.next_batch()
        accuracy = sess.run(acc, feed_dict={video_placeholder:image,
                                            audio_placeholder:audio,
                                            label_placeholder:label,
                                            score_placeholder:score})
        print('->',accuracy)
        final_acc+=accuracy
        c += 1
    print(final_acc/c)


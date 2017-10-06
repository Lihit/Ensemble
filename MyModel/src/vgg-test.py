import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from detector import Detector
from util import load_image

import skimage.io
import matplotlib.pyplot as plt

import os
#import ipdb

testset_path = '../data/caltech/test.pickle'
label_dict_path = '../data/caltech/label_dict.pickle'

weight_path = '../models/mymodel/caffe_layers_value.pickle'
model_path = '../models/mymodel/model-4'

batch_size = 1

testset = pd.read_pickle( testset_path )
label_dict = pd.read_pickle( label_dict_path )
n_labels = len( label_dict )

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector( weight_path, n_labels )
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference( images_tf )
classmap = detector.get_classmap( labels_tf, conv6 )

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )
n_correct = 0
n_data = 0
for start, end in zip(
    range( 0, len(testset)+batch_size, batch_size),
    range(batch_size, len(testset)+batch_size, batch_size)):

    current_data = testset[start:end]
    current_image_paths = current_data['image_path'].values
    current_images = np.array(map(lambda x: load_image(x), current_image_paths))

    good_index = np.array(map(lambda x: x is not None, current_images))

    current_data = current_data[good_index]
    current_image_paths = current_image_paths[good_index]
    current_images = np.stack(current_images[good_index])
    current_labels = current_data['label'].values
    current_label_names = current_data['label_name'].values
    
    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
                })

    label_predictions = output_val.argmax( axis=1 )
    _,top_3=tf.nn.top_k(output_val,3)
    val_top_3=sess.run(top_3,
                feed_dict={
                    images_tf: current_images
                    })
    classmap_vals = sess.run(
            classmap,
            feed_dict={
                labels_tf: label_predictions,
                conv6: conv6_val
                })

    classmap_answer = sess.run(
            classmap,
            feed_dict={
                labels_tf: current_labels,
                conv6: conv6_val
                })

    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer)
    acc=0
    for j,row in enumerate(val_top_3):
        if current_labels[j] in row:
            acc+=1
    #acc = (label_predictions == current_labels).sum()
    n_correct += acc
    n_data += len(current_data)
    for vis, ori,ori_path, l_name,predictIndex in zip(classmap_vis, current_images, current_image_paths, current_label_names,label_predictions):
        print 'ture label:'+l_name+'    predicted label:'+ label_dict.index[predictIndex]
        img_origin=plt.imread(ori_path)
        plt.subplot(121)
        plt.imshow(img_origin)
        plt.subplot(122)
        plt.imshow( ori )
        plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
        #plt.subplot(122)
        #plt.imshow( vis)
        plt.show()

        #vis_path = '../results/'+ ori_path.split('/')[-1]
        #vis_path_ori = '../results/'+ori_path.split('/')[-1].split('.')[0]+'.ori.jpg'
        #skimage.io.imsave( vis_path, vis )
        #skimage.io.imsave( vis_path_ori, ori )
acc_all = n_correct / float(n_data)
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print 'testDataNum:' + str(n_data) + '\tcorrectNum:'+str(n_correct)+'\tacc:' + str(acc_all)
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

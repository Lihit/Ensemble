import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from detector import Detector
from util import load_image

import skimage.io
import matplotlib.pyplot as plt

import os
import json
#import ipdb
resultsList=[]
json_filepath='result.json'
fp=open(json_filepath,'w')
weight_path = '../models/mymodel/caffe_layers_value.pickle'
model_path = '../models/mymodel/model-4'
dataset_path = '/home/tensor/tensor/scene/DataSet/test'
current_image_paths=[]
for imageName in os.listdir(dataset_path):
    current_image_paths.append(os.path.join(dataset_path,imageName))

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")

detector = Detector( weight_path, 80 )
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference( images_tf )
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore( sess, model_path )
for oneImagePath in current_image_paths:
    print(oneImagePath)
    current_image = np.reshape(np.array(load_image(oneImagePath)),(1,224,224,3))
    if current_image is None:
        continue
    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_image
                })
    _,top_3=tf.nn.top_k(output_val,3)
    val_top_3=sess.run(top_3,
                feed_dict={
                    images_tf: current_image
                    })
    imageResultDict={}
    imageResultDict['image_id']=str(oneImagePath.split('/')[-1])
    imageResultDict['label_id']=[int(x) for x in val_top_3[0]]
    resultsList.append(imageResultDict)
json.dump(resultsList,fp)
fp.close()

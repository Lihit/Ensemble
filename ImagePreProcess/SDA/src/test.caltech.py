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

if not os.path.exists('results/'):
    os.mkdir('results/')
json_filepath='results/result-model4-test.json'
resultsList=[]
testset_path = '../data/mydata/test.pickle'
label_dict_path ='../data/mydata/label_dict.pickle'

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
n_correct_top3= 0
n_correct_top5= 0
n_data = 0
f_test = open('results/testlog.txt', 'w')
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
    _,top_3=tf.nn.top_k(output_val,10)
    val_top_3=sess.run(top_3,
                feed_dict={
                    images_tf: current_images
                    })
    acc_top3=0
    acc_top5=0
    for j,row in enumerate(val_top_3):
        imageResultDict={}
        imageResultDict['image_id']=str(current_image_paths[j].split('/')[-1])
        imageResultDict['label_id']=[int(x) for x in row]
        resultsList.append(imageResultDict)
        if current_labels[j] in row[:3]:
            acc_top3+=1
        if current_labels[j] in row:
            acc_top5+=1
    #acc = (label_predictions == current_labels).sum()
    n_correct_top3 += acc_top3
    n_correct_top5 += acc_top5
    n_data += len(current_data)
    for ori_path,l_name,predictIndex in zip(current_image_paths,current_label_names,label_predictions):
        f_test.write('image_path: '+ori_path+'\tture label:'+l_name+'\tpredicted label:'+ label_dict.index[predictIndex]+'\n')
        print 'ture label:'+l_name+'    predicted label:'+ label_dict.index[predictIndex]
        #img_origin=plt.imread(ori_path)
        #plt.subplot(121)
        #plt.imshow(img_origin)
        #plt.subplot(122)
        #plt.imshow( ori )
        #plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
        #plt.subplot(122)
        #plt.imshow( vis)
        #plt.show()

        #vis_path = '../results/'+ ori_path.split('/')[-1]
        #vis_path_ori = '../results/'+ori_path.split('/')[-1].split('.')[0]+'.ori.jpg'
        #skimage.io.imsave( vis_path, vis )
        #skimage.io.imsave( vis_path_ori, ori )
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print 'testDataNum:' + str(n_data) + '\tcorrectNum_top3:'+str(n_correct_top3)+'\tacc:' + str(n_correct_top3 / float(n_data))
print 'testDataNum:' + str(n_data) + '\tcorrectNum_top5:'+str(n_correct_top5)+'\tacc:' + str(n_correct_top5 / float(n_data))
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
with open(json_filepath,'w') as fp:
    json.dump(resultsList,fp)
f_test.close()

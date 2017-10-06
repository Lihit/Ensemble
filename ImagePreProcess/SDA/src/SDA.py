import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random
import cv2
from detector import Detector
from util import load_image
import skimage.io
import matplotlib.pyplot as plt


def getImagePath(ImagePath):
    ImagePaths = []
    if not os.path.exists(ImagePath):
        return ImagePaths
    else:
        if os.path.isfile(ImagePath) and ImagePath.endswith('.jpg') and os.path.basename(ImagePath).count('_') == 1:
            ImagePaths.append(ImagePath)
        elif os.path.isdir(ImagePath):
            for subfile in os.listdir(ImagePath):
                ImagePaths.extend(getImagePath(
                    os.path.join(ImagePath, subfile)))
    return ImagePaths


def cropImage(heatImageOri, originImage, originImagePath):
    heatImageInput = heatImageOri.copy()
    imgOri = originImage.copy()
    if not os.path.exists(originImagePath):
        return
    if len(heatImageInput.shape) == 3:
        heatImage = cv2.cvtColor(heatImageInput, cv2.COLOR_BGR2GRAY)
    else:
        heatImage = heatImageInput
    imgOri_h, imgOri_w, _ = imgOri.shape
    heatImage_h, heatImage_w = heatImage.shape
    maxIndex = np.argmax(heatImage)
    i, j = maxIndex / heatImage_w, maxIndex % heatImage_w
    crop_h, crop_w = int((80 + random.randint(0, 20)) * (imgOri_h / heatImage_h)
                         ), int((80 + random.randint(0, 20)) * (imgOri_w / heatImage_w))
    crop_i, crop_j = int(i * (imgOri_h / heatImage_h)
                         ), int(j * (imgOri_w / heatImage_w))
    xmin_crop = int(crop_i - crop_h / 3) if int(crop_i - crop_h / 3) > 0 else 0
    ymin_crop = int(crop_j - crop_w / 3) if int(crop_j - crop_w / 3) > 0 else 0
    xmax_crop = int(crop_i + crop_h * 2 / 3) if int(crop_i +
                                                    crop_h * 2 / 3) < imgOri_h else imgOri_h - 1
    ymax_crop = int(crop_j + crop_w * 2 / 3) if int(crop_j +
                                                    crop_w * 2 / 3) < imgOri_w else imgOri_w - 1
    cropImageRet = imgOri[xmin_crop:xmax_crop, ymin_crop:ymax_crop]
    cv2.imwrite(os.path.join(os.path.dirname(originImagePath), os.path.basename(
        originImagePath).replace('.jpg', '_1.jpg')), cropImageRet)
    return cropImageRet


def SDA(testImagePath):
    label_dict_path = '../data/caltech/label_dict.pickle'
    weight_path = '../models/mymodel/caffe_layers_value.pickle'
    model_path = '../models/mymodel/model-5'
    testset = np.array(getImagePath(testImagePath))
    label_dict = pd.read_pickle(label_dict_path)
    n_labels = len(label_dict)
    images_tf = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
    labels_tf = tf.placeholder(tf.int64, [None], name='labels')
    detector = Detector(weight_path, n_labels)
    c1, c2, c3, c4, conv5, conv6, gap, output = detector.inference(images_tf)
    classmap = detector.get_classmap(labels_tf, conv6)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    n_correct = 0
    n_data = 0
    batch_size = 1
    for start, end in zip(
            range(0, len(testset), batch_size),
            range(batch_size, len(testset) + batch_size, batch_size)):
        current_image_paths = testset[start:end]
        current_images = np.array(
            map(lambda x: load_image(x), current_image_paths))
        good_index = np.array(map(lambda x: x is not None, current_images))
        current_image_paths = current_image_paths[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = np.array(
            map(lambda x: int(os.path.basename(x).split('_')[0]), current_image_paths))
        current_label_names = np.array(
            map(lambda x: label_dict.index[int(os.path.basename(x).split('_')[0])], current_image_paths))
        conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
            })

        label_predictions = output_val.argmax(axis=1)
        _, top_3 = tf.nn.top_k(output_val, 3)
        val_top_3 = sess.run(top_3,
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

        classmap_vis = map(lambda x: (
            (x - x.min()) / (x.max() - x.min())), classmap_answer)
        acc = 0
        for j, row in enumerate(val_top_3):
            if current_labels[j] in row:
                acc += 1
        #acc = (label_predictions == current_labels).sum()
        n_correct += acc
        n_data += len(current_labels)
        for vis, ori, ori_path, l_name, predictIndex in zip(classmap_vis, current_images, current_image_paths, current_label_names, label_predictions):
            print 'ture label:' + l_name + '    predicted label:' + label_dict.index[predictIndex]
            print(ori_path)
            img_origin = plt.imread(ori_path)
            plt.subplot(131)
            plt.imshow(img_origin)
            plt.subplot(132)
            plt.imshow(ori)
            plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
            plt.subplot(133)
            cropImageRet = cropImage(vis, ori, ori_path)
            plt.imshow(cropImageRet)
            plt.show()
    acc_all = n_correct / float(n_data)
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print 'testDataNum:' + str(n_data) + '\tcorrectNum:' + str(n_correct) + '\tacc:' + str(acc_all)
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


if __name__ == '__main__':
    SDA('ImageResult/')

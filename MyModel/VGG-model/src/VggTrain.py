import tensorflow as tf
import numpy as np
import pandas as pd

from VggNet import Detector
from util import load_image
import os
import csv


def getImagePaths(dataset_path):
    image_paths_per_label = []
    for one_dir in os.listdir(dataset_path):
        listtmp = []
        for one_file in os.listdir(os.path.join(dataset_path, one_dir)):
            if one_file.count('_') != 1:
                listtmp.append(os.path.join(
                    dataset_path, one_dir, one_file))
        image_paths_per_label.append(listtmp)
    return image_paths_per_label


def main():
    weight_path = '/home/tensor/tensor/wsg/Weakly_detector/models/mymodel/caffe_layers_value.pickle'
    model_path = '../models/mymodel/'
    pretrained_model_path = '/home/tensor/tensor/wsg/Weakly_detector/models/mymodel/model-4'
    n_epochs = 8
    init_learning_rate = 0.01
    weight_decay_rate = 0.0005
    momentum = 0.9
    batch_size = 60

    dataset_path = '/home/tensor/tensor/scene/DataSet/train'

    caltech_path = '../data/mydata'
    trainset_path = '../data/mydata/train.pickle'
    testset_path = '../data/mydata/validation.pickle'
    label_dict_path = '../data/mydata/label_dict.pickle'

    if not os.path.exists(trainset_path):
        if not os.path.exists(caltech_path):
            os.makedirs(caltech_path)
        labels = []
        label_names = []
        f = open('scene_classes.csv', 'r')
        r = csv.reader(f)
        for row in r:
            labels.append(row[0])
            label_names.append(row[-1])
        label_dict = pd.Series(labels, index=label_names)
        n_labels = len(label_dict)

        image_paths_train = np.hstack(getImagePaths(
            '/home/tensor/tensor/scene/DataSet/train'))
        image_paths_test = np.hstack(getImagePaths(
            '/home/tensor/tensor/scene/DataSet/validation'))

        trainset = pd.DataFrame({'image_path': image_paths_train})
        testset = pd.DataFrame({'image_path': image_paths_test})

        trainset = trainset[trainset['image_path'].map(
            lambda x: x.endswith('.jpg'))]
        trainset['label'] = trainset['image_path'].map(
            lambda x: int(x.split('/')[-2].split('.')[0]))
        trainset['label_name'] = trainset['image_path'].map(
            lambda x: x.split('/')[-2].split('.')[1])

        testset = testset[testset['image_path'].map(
            lambda x: x.endswith('.jpg'))]
        testset['label'] = testset['image_path'].map(
            lambda x: int(x.split('/')[-2].split('.')[0]))
        testset['label_name'] = testset['image_path'].map(
            lambda x: x.split('/')[-2].split('.')[1])

        label_dict.to_pickle(label_dict_path)
        trainset.to_pickle(trainset_path)
        testset.to_pickle(testset_path)
    else:
        trainset = pd.read_pickle(trainset_path)
        testset = pd.read_pickle(testset_path)
        label_dict = pd.read_pickle(label_dict_path)
        n_labels = len(label_dict)

    learning_rate = tf.placeholder(tf.float32, [])
    images_tf = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
    labels_tf = tf.placeholder(tf.int64, [None], name='labels')

    detector = Detector(weight_path, n_labels)

    p1, p2, p3, p4, conv5, conv6, fc6, output = detector.inference(images_tf)
    loss_tf = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels_tf))

    weights_only = filter(lambda x: x.name.endswith('W:0'),
                          tf.trainable_variables())
    weight_decay = tf.reduce_sum(
        tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
    loss_tf += weight_decay

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=50)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    grads_and_vars = optimizer.compute_gradients(loss_tf)
    grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[
                         1].name or 'fc6' in gv[1].name) else (gv[0] * 0.1, gv[1]), grads_and_vars)
    #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)
    tf.initialize_all_variables().run()

    if pretrained_model_path:
        print "Pretrained"
        saver.restore(sess, pretrained_model_path)

    testset.index = range(len(testset))
    # testset = testset.ix[np.random.permutation( len(testset) )]#[:1000]
    #trainset2 = testset[1000:]
    #testset = testset[:1000]

    #trainset = pd.concat( [trainset, trainset2] )
    # We lack the number of training set. Let's use some of the test images

    f_log_acc = open('../results/log.sceneAcc.txt', 'w')
    f_log_loss = open('../results/log.sceneLoss.txt', 'w')

    iterations = 0
    loss_list = []
    for epoch in range(n_epochs):

        trainset.index = range(len(trainset))
        trainset = trainset.ix[np.random.permutation(len(trainset))]

        for start, end in zip(
                range(0, len(trainset) + batch_size, batch_size),
                range(batch_size, len(trainset) + batch_size, batch_size)):

            current_data = trainset[start:end]
            current_image_paths = current_data['image_path'].values
            current_images = np.array(
                map(lambda x: load_image(x), current_image_paths))

            good_index = np.array(map(lambda x: x is not None, current_images))

            current_data = current_data[good_index]
            current_images = np.stack(current_images[good_index])
            current_labels = current_data['label'].values

            _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                })

            loss_list.append(loss_val)

            iterations += 1
            if iterations % 5 == 0:
                print "======================================"
                print "Epoch", epoch, "Iteration", iterations
                print "Processed", start, '/', len(trainset)

                label_predictions = output_val.argmax(axis=1)
                acc = (label_predictions == current_labels).sum()

                print "Accuracy:", acc, '/', len(current_labels)
                print "Training Loss:", np.mean(loss_list)
                print "\n"
                f_log_loss.write(" Epoch:" + str(epoch) + " Iteration:" + str(
                    iterations) + " Training Loss:" + str(np.mean(loss_list)) + '\n')
                loss_list = []

        n_correct = 0
        n_data = 0
        for start, end in zip(
                range(0, len(testset) + batch_size, batch_size),
                range(batch_size, len(testset) + batch_size, batch_size)
        ):
            current_data = testset[start:end]
            current_image_paths = current_data['image_path'].values
            current_images = np.array(
                map(lambda x: load_image(x), current_image_paths))

            good_index = np.array(map(lambda x: x is not None, current_images))

            current_data = current_data[good_index]
            current_images = np.stack(current_images[good_index])
            current_labels = current_data['label'].values

            output_vals = sess.run(
                output,
                feed_dict={images_tf: current_images})

            label_predictions = output_vals.argmax(axis=1)
            acc = (label_predictions == current_labels).sum()

            n_correct += acc
            n_data += len(current_data)

        acc_all = n_correct / float(n_data)
        f_log_acc.write('epoch:' + str(epoch) + '\tacc:' + str(acc_all) + '\n')
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print 'epoch:' + str(epoch) + '\tacc:' + str(acc_all) + '\n'
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        saver.save(sess, os.path.join(
            model_path, 'model'), global_step=0)
        init_learning_rate *= 0.99
    f_log_acc.close()
    f_log_loss.close()


if __name__ == '__main__':
    main()
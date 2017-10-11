import os
import csv
import numpy as np
import pandas as pd

def getImagePaths(dataset_path, flag):
    image_paths_per_label = []
    for one_dir in os.listdir(dataset_path):
        listtmp = []
        for one_file in os.listdir(os.path.join(dataset_path, one_dir)):
            if one_file.count('_') != 1 or flag:
                listtmp.append(os.path.join(
                    dataset_path, one_dir, one_file))
        image_paths_per_label.append(listtmp)
    return image_paths_per_label

def main():
    testset_path = '../data/mydata/test.pickle'
    testImagePath = getImagePaths(
                '/home/tensor/tensor/scene/DataSet/test', 1)
    image_paths_test = np.hstack(map(lambda x: x, testImagePath))
    testset = pd.DataFrame({'image_path': image_paths_test})
    testset = testset[testset['image_path'].map(
                lambda x: x.endswith('.jpg'))]
    testset.to_pickle(testset_path)

if __name__ == '__main__':
    main()
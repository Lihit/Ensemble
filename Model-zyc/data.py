import cv2
import numpy as np
import json
import os
import random as rd

class DataGenerator():

    # yc
    def __init__(self, train_image_dir, validate_image_dir, input_size=299):
        def generate_img_dict(path):
            img_cnt_dict = {}
            img_path_dict = {}
            for calss_name in os.listdir(path):
                if calss_name.split(".")[0] != "":
                    class_idx = int(calss_name.split(".")[0])
                    img_list = os.listdir(path + calss_name)
                    img_cnt_dict[class_idx] = int(len(img_list) / 4)
                    img_path_dict[class_idx] = [path + calss_name + "/" + str(class_idx) + "_" + str(i + 1) + ".jpg" 
                                                for i in range(img_cnt_dict[class_idx])]
            return img_cnt_dict, img_path_dict
    
        print 'Loading train and validate samples...'
        self.train_image_dir = train_image_dir
        self.validate_image_dir = validate_image_dir
        
        self.train_img_cnt_dict, self.trian_img_path_dict = generate_img_dict(self.train_image_dir)
        self.validate_img_cnt_dict, self.validate_img_path_dict = generate_img_dict(self.validate_image_dir)

        self.train_count = reduce(lambda x,y:x+y, [self.train_img_cnt_dict[l] for l in self.train_img_cnt_dict])
        self.validate_count = reduce(lambda x,y:x+y, [self.validate_img_cnt_dict[l] for l in self.validate_img_cnt_dict])

        self.input_size = input_size
        self.num_class = len(self.train_img_cnt_dict)
        print 'train samples {}, validate samples {}'.format(self.train_count, self.validate_count)

    # yc
    def generate_img_dict(self, path):
        img_cnt_dict = {}
        img_path_dict = {}
        for calss_name in os.listdir(path):
            if calss_name.split(".")[0] != "":
                class_idx = int(calss_name.split(".")[0])
                img_list = os.listdir(path + calss_name)
                img_cnt_dict[class_idx] = int(len(img_list) / 4)
                img_path_dict[class_idx] = [path + calss_name + "/" + str(class_idx) + "_" + str(i + 1) + ".jpg" 
                                            for i in range(img_cnt_dict[class_idx])]
        return img_cnt_dict, img_path_dict


    # yc 
    def load_image_from_file(self, img_path):
        img = cv2.imread(img_path)
        return img


    # yc
    def generate_batch_train_samples(self, img_cnt_dict, img_path_dict, batch_size=32):
        
        num_class = len(img_cnt_dict)
        
        img_idx = [0] * num_class
        idx_class = 0

        batch_x = np.zeros((batch_size, self.input_size, self.input_size, 3))
        batch_y = np.zeros(batch_size)
        
        while True:
            for i in xrange(batch_size):
                if img_idx[idx_class] == img_cnt_dict[idx_class]:
                    rd.shuffle(img_path_dict[idx_class])
                    img_idx[idx_class] = 0
                image = self.load_image_from_file(img_path_dict[idx_class][img_idx[idx_class]])
                image = cv2.resize(image, (self.input_size, self.input_size))
                label = idx_class
                
                batch_x[i] = image
                batch_y[i] = label
    
                img_idx[idx_class] += 1
                idx_class = (idx_class + 1) % num_class
            yield batch_x, batch_y
    
    # yc
    def generate_validate_samples(self, img_cnt_dict, img_path_dict, batch_size=32):

        num_class = len(img_cnt_dict)
        
        img_idx = [0] * num_class
        idx_class = 0
        index = 0
        
        batch_x = np.zeros((batch_size, self.input_size, self.input_size, 3))
        batch_y = np.zeros(batch_size)    

        while index < self.validate_count:
            for i in xrange(batch_size):
                if img_idx[idx_class] == img_cnt_dict[idx_class]:
                    rd.shuffle(img_path_dict[idx_class])
                    img_idx[idx_class] = 0
                image = self.load_image_from_file(img_path_dict[idx_class][img_idx[idx_class]])
                image = cv2.resize(image, (self.input_size, self.input_size))
                label = idx_class
                
                batch_x[i] = image
                batch_y[i] = label
    
                img_idx[idx_class] += 1
                idx_class = (idx_class + 1) % num_class

                index += 1
                if index == self.validate_count:
                    break
            yield batch_x, batch_y

    # yc
    def get_validate_sample_count(self):
        return self.validate_count

    # yc
    def get_train_sample_count(self):
        return self.train_count

import tensorflow as tf
from data import DataGenerator
from model_factory import ModelFactory

def main():
    model_dir = "/home/tensor/tensor/scene/DataSet/checkpoints/"
    train_image_dir = "/home/tensor/tensor/scene/DataSet/train/"
    validate_image_dir = "/home/tensor/tensor/scene/DataSet/validation/"
    pretrained_model_path = "/home/tensor/tensor/scene/DataSet/pre_trained/inception_resnet_v2.ckpt"
    datagen = DataGenerator(train_image_dir, validate_image_dir)
    model = ModelFactory(datagen, net='INCEPTION_RESNET_V2', model_dir=model_dir, fine_tune=True, 
        pretrained_path=pretrained_model_path)
    with tf.Session() as session:
        model.train(session)

if __name__ == '__main__':
    main()

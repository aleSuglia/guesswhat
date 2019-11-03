#!/usr/bin/env python
import os
from argparse import ArgumentParser
from os import listdir
from time import time

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
from PIL import Image
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument("-image_dir", help="Path to the directory containing the images")
parser.add_argument("-output_dir", help="ResNet features output directory where the features metadata will be stored")
parser.add_argument("-model_ckpt", default="data/vgg.ckpt", help="Path to the VGG-16 Net model checkpoint")


def get_splits(image_dir):
    dirs = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]

    if not dirs:
        return ['train', 'val', 'test']

    return dirs


def extract_features(
        images_placeholder,
        image_dir,
        split,
        ft_output,
        network_ckpt,
        out_dir):

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, network_ckpt)
        img_list = listdir(os.path.join(image_dir, split))

        print("Load dataset -> set: {}".format(split))

        no_images = len(img_list)

        ############################
        #  CREATE FEATURES
        ############################
        print("Start computing image features...")
        filepath = os.path.join(out_dir, "{}_features.h5".format(split))
        with h5py.File(filepath, 'w') as f:
            ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]
            ft_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
            idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)

            for i in tqdm(range(len(img_list))):
                image_filepath = os.path.join(image_dir, split, img_list[i])
                image_tensor = Image.open(image_filepath).convert('RGB')

                feat = sess.run(ft_output, feed_dict={images_placeholder: image_tensor})

                # Store dataset
                ft_dataset[i] = feat

                idx2img[i] = img_list[i]

                print("Start dumping file: {}".format(filepath))
            print("Finished dumping file: {}".format(filepath))

    print("Done!")


def main(args):
    start = time()
    print('Start')
    splits = get_splits(args.image_dir)
    img_size = 224
    images_placeholder = tf.placeholder(tf.float32, [None, img_size, img_size, 3], name='image')

    _, end_points = vgg.vgg_16(images_placeholder, is_training=False, dropout_keep_prob=1.0)
    ft_name = os.path.join("vgg_16", args.feature_name)
    ft_output = end_points[ft_name]

    for split in splits:
        extract_features(
            images_placeholder=images_placeholder,
            image_dir=os.path.join(args.image_dir, split),
            ft_output=ft_output,
            out_dir=args.output_dir,
            split=split,
            network_ckpt=args.model_ckpt)

    print('Image Features extracted.')
    print('Time taken: ', time() - start)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

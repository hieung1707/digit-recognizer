import struct
import cv2
import numpy as np
import pickle
from os import path

# file names
train_img_name = 'train-images.idx3-ubyte'
train_label_name = 'train-labels.idx1-ubyte'
train_pkl_name = 'train-data.pkl'
test_img_name = 't10k-images.idx3-ubyte'
test_label_name = 't10k-labels.idx1-ubyte'
test_pkl_name = 'test-data.pkl'

# path
data_dir = 'hand_written_digits'
train_img_path = '{}/{}'.format(data_dir, train_img_name)
train_label_path = '{}/{}'.format(data_dir, train_label_name)

test_img_path = '{}/{}'.format(data_dir, test_img_name)
test_label_path = '{}/{}'.format(data_dir, test_label_name)


def get_image(path):
    images = []
    img_num = 0
    with open(path, "rb") as f:
        try:
            byte = f.read(4)
            magic_num = np.fromstring(byte, dtype='>u4')
            byte = f.read(4)
            img_num = np.fromstring(byte, dtype='>u4')
            byte = f.read(4)
            rows = np.fromstring(byte, dtype='>u4')
            byte = f.read(4)
            cols = np.fromstring(byte, dtype='>u4')
            for index in range(img_num):
                image = np.zeros((28, 28), np.uint8)
                print 'data: {}/{}'.format(index, img_num)
                for i in range(rows):
                    for j in range(cols):
                        byte = f.read(1)
                        image[i][j] = np.fromstring(byte, dtype='>u1')
                images.append(image)
        except Exception, e:
            print e
            # while byte != "":
            # pixel = np.frombuffer(byte, np.uint8)
            # byte_array.append(byte)
            # data += bytearray(byte).decode(encoding='utf-16')
            # Do stuff with byte.
            #     byte = f.read(1)
        finally:
            f.close()
        return img_num, images


def get_labels(path):
    labels = []
    labels_num = 0
    with open(path, "rb") as f:
        try:
            byte = f.read(4)
            magic_num = np.fromstring(byte, dtype='>u4')
            byte = f.read(4)
            labels_num = np.fromstring(byte, dtype='>u4')
            for _ in range(labels_num):
                byte = f.read(1)
                label = np.fromstring(byte, dtype='>u1')
                labels.append(label)
        except Exception, e:
            print e
            # while byte != "":
            # pixel = np.frombuffer(byte, np.uint8)
            # byte_array.append(byte)
            # data += bytearray(byte).decode(encoding='utf-16')
            # Do stuff with byte.
            #     byte = f.read(1)
        finally:
            f.close()
            return labels_num, labels


# images = get_image(train_img_path)
# labels = get_labels(train_label_path)
# print 'finished loading dataset'
# training_set = {}
# for i in range(10):
#     training_set[i] = []
# for i in range(60000):
#     label = labels[i]
#     training_set[label[0]].append(images[i])
#
# pkl_file = open('train_data.pkl', 'wb')
# pickle.dump(training_set, pkl_file)
# pkl_file.close()

def get_data_from_pkl(type):
    data = {}
    types = {
        "train": [train_pkl_name,
                  train_img_path,
                  train_label_path],
        "test": [test_pkl_name,
                 test_img_path,
                 test_label_path]
    }
    pkl_name = types[type][0]
    img_path = types[type][1]
    label_path = types[type][2]

    if path.exists(pkl_name):
        print 'loading data from pickle file'
        file = open(pkl_name, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        print 'loading data from binaries'
        n_imgs, imgs = get_image(img_path)
        n_labels, labels = get_labels(label_path)
        if n_imgs != n_labels:
            print 'number of samples mismatch'
            return {}
        for i in range(n_imgs):
            data[i] = [imgs[i], labels[i]]
        file = open(pkl_name, 'wb')
        pickle.dump(data, file)
        file.close()
    return data

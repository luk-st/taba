import os
import pickle

import imageio
import numpy as np
from tqdm import tqdm

os.chdir("<PATH_TO_IMAGNET>/imagenet64")


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo)
    return dict


def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, "val_data")

    d = unpickle(test_file)
    x = d["data"]
    y = d["labels"]
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i - 1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(X_test=x, Y_test=y.astype("int64"))


def load_data_batch(data_folder, idx, img_size=64):
    data_file = os.path.join(data_folder, "train_data_batch_")

    d = unpickle(data_file + str(idx))
    x = d["data"]
    y = d["labels"]
    mean_image = d["mean"]

    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(X_train=X_train, Y_train=Y_train.astype("int64"), mean=mean_image)


def get_images(f, img_size=32):

    d = unpickle(f)
    x = d["data"]
    img_size = 64
    img_size2 = 64 * 64
    x = np.dstack((x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    print(x.shape)
    return x, d["labels"]


# load train batch data and save into png file
# load validation batch data and decompress and save
data_folder = "."
num_classes = 1000
test_file = os.path.join(data_folder, "val_data")
val_data, labels = get_images(test_file, 64)
k = 1
for i, e in enumerate(val_data):
    imageio.imwrite("imagenet64/val/%05d_%08d.png" % (labels[i], i), e)
for i in tqdm(list(range(1, 11)), "Iterating batches"):
    f = os.path.join(data_folder, "train_data_batch_") + str(i)
    train_data, labels = get_images(f, 64)
    for j, e in tqdm(enumerate(train_data), total=train_data.shape[0], desc="Iterating samples"):
        imageio.imwrite("imagenet64/train/%05d_%08d.png" % (labels[j], k), e)
        k += 1

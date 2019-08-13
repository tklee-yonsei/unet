from __future__ import print_function

import glob
import os

import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjust_data(img, label, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
        new_label = np.zeros(label.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in label and convert it into one-hot vector
            # index = np.where(label == i)
            # index_label = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i)
            #   if (len(label.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_label[index_label] = 1
            new_label[label == i, i] = 1
        new_label = np.reshape(new_label, (new_label.shape[0],
                                           new_label.shape[1] * new_label.shape[2],
                                           new_label.shape[3])) \
            if flag_multi_class else np.reshape(new_label,
                                                (new_label.shape[0] * new_label.shape[1], new_label.shape[2]))
        label = new_label
    elif np.max(img) > 1:
        img = img / 255
        label = label / 255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return img, label


def image_label_set_generator(batch_size, path, image_folder, label_folder, image_data_generator_dict,
                              image_color_mode="grayscale", label_color_mode="grayscale", image_save_prefix="image",
                              label_save_prefix="label", flag_multi_class=False, num_class=2, save_to_dir=None,
                              target_size=(256, 256), seed=1):
    """
    can generate image and label at the same time
    use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    image_data_generator = ImageDataGenerator(**image_data_generator_dict)
    label_data_generator = ImageDataGenerator(**image_data_generator_dict)
    image_generator = image_data_generator.flow_from_directory(
        path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    label_generator = label_data_generator.flow_from_directory(
        path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)
    train_generators = zip(image_generator, label_generator)
    for (img, label) in train_generators:
        img, label = adjust_data(img, label, flag_multi_class, num_class)
        yield (img, label)


def test_generator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def test_via_path_generator(test_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    filenames = os.listdir(test_path)
    filtered_only_png = list(filter(lambda f: f.endswith('.png'), filenames))
    for file in filtered_only_png:
        img = io.imread(os.path.join(test_path, file), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def generate_train_numpy(image_path, label_path, flag_multi_class=False, num_class=2, image_prefix="image",
                         label_prefix="label", image_as_gray=True, label_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    label_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        label = io.imread(item.replace(image_path, label_path).replace(image_prefix, label_prefix),
                          as_gray=label_as_gray)
        label = np.reshape(label, label.shape + (1,)) if label_as_gray else label
        img, label = adjust_data(img, label, flag_multi_class, num_class)
        image_arr.append(img)
        label_arr.append(label)
    image_arr = np.array(image_arr)
    label_arr = np.array(label_arr)
    return image_arr, label_arr


def label_visualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(save_path, numpy_file, flag_multi_class=False, num_class=2):
    for i, item in enumerate(numpy_file):
        img = label_visualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


def save_result_with_name(save_path, numpy_file, name_list, flag_multi_class=False, num_class=2):
    for i, item in enumerate(numpy_file):
        img = label_visualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, name_list[i]), img)

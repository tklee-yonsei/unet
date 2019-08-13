import os
import sys

from keras.callbacks import ModelCheckpoint

from data import image_label_set_generator
from model import unet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    data_set_name = sys.argv[1]

    print("--- Training ---")
    training_folder = os.path.join("data", data_set_name, "train")
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    train_generator = image_label_set_generator(2, training_folder, 'image', 'label', data_gen_args, save_to_dir=None)
    model = unet()
    model_checkpoint = ModelCheckpoint(data_set_name + ".hdf5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    # print("--- Validation ---")

    print("--- Test ---")
    test_folder = os.path.join("data", data_set_name, "test")
    test_label_data_folder = os.path.join(test_folder, "label")
    test_label_folder_files = os.listdir(test_label_data_folder)
    only_png_files_in_test_label_folder = list(filter(lambda f: f.endswith('.png'), test_label_folder_files))
    test_generator = image_label_set_generator(2, test_folder, 'image', 'label', data_gen_args, save_to_dir=None)
    scores = model.evaluate_generator(test_generator, len(only_png_files_in_test_label_folder))
    for index, metrics_name in enumerate(model.metrics_names):
        print("%s: %.2f%%" % (model.metrics_names[index], scores[index] * 100))

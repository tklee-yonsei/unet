import os
import sys

from keras.callbacks import ModelCheckpoint

from data import image_label_set_generator
from model import unet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    data_set_name = sys.argv[1]
    test_result_folder = sys.argv[2]

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    train_generator = image_label_set_generator(2, "data/" + data_set_name + "/train", 'image', 'label', data_gen_args,
                                                save_to_dir=None)

    print("--- Training ---")
    model = unet()
    model_checkpoint = ModelCheckpoint(data_set_name + ".hdf5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(train_generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    # print("--- Validation ---")

    print("--- Test ---")
    files = os.listdir("data/" + data_set_name + "/test")
    filtered_only_png = list(filter(lambda f: f.endswith('.png'), files))
    test_generator = train_generator(2, "data/" + data_set_name + "/test", 'image', 'label', data_gen_args,
                                     save_to_dir=None)
    scores = model.evaluate_generator(test_generator, len(filtered_only_png))
    for index, metrics_name in enumerate(model.metrics_names):
        print("%s: %.2f%%" % (model.metrics_names[index], scores[index] * 100))

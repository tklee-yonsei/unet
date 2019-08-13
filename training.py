import os
import sys

from keras.callbacks import ModelCheckpoint

from data import train_generator
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
    my_generator = train_generator(2, "data/" + data_set_name + "/train", 'image', 'label', data_gen_args,
                                   save_to_dir=None)

    print("--- Training ---")
    model = unet()
    model_checkpoint = ModelCheckpoint(data_set_name + ".hdf5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(my_generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    print("--- Validation ---")

    print("--- Test ---")
    my_test_generator = train_generator(2, "data/" + data_set_name + "/test2", 'image', 'label', data_gen_args,
                                        save_to_dir=None)
    scores = model.evaluate_generator(my_test_generator, )
    for index, metrics_name in enumerate(model.metrics_names):
        print("%s: %.2f%%" % (model.metrics_names[index], scores[index] * 100))

    # print("--- Predict ---")
    # test_generator = test_via_path_generator("data/" + data_set_name + "/test")
    # results = model.predict_generator(test_generator, 30, verbose=1)
    # if not os.path.exists(test_result_folder):
    #     os.makedirs(test_result_folder)
    # save_result(test_result_folder, results)
    # files = os.listdir("data/" + data_set_name + "/test")
    # for file in files:
    #     shutil.copy("data/" + data_set_name + "/test/" + file, test_result_folder)

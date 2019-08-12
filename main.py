import os
import shutil
import sys

from keras.callbacks import ModelCheckpoint

from data import train_generator, test_generator, save_result
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
    myGene = train_generator(2, "data/" + data_set_name + "/train", 'image', 'label', data_gen_args, save_to_dir=None)

    model = unet()
    model_checkpoint = ModelCheckpoint(data_set_name + ".hdf5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    testGene = test_generator("data/" + data_set_name + "/test")
    results = model.predict_generator(testGene, 30, verbose=1)

    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)
    save_result(test_result_folder, results)

    files = os.listdir("data/" + data_set_name + "/test")
    for file in files:
        shutil.copy("data/" + data_set_name + "/test/" + file, test_result_folder)

import os
import sys

from keras.callbacks import ModelCheckpoint

from data import trainGenerator, testGenerator, saveResult
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
    myGene = trainGenerator(2, "data/" + data_set_name + "/train", 'image', 'label', data_gen_args, save_to_dir=None)

    model = unet()
    model_checkpoint = ModelCheckpoint(data_set_name + ".hdf5", monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    testGene = testGenerator("data/" + data_set_name + "/test")
    results = model.predict_generator(testGene, 30, verbose=1)

    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)
    saveResult(test_result_folder, results)

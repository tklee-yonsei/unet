import os
import pathlib
import shutil
import sys

from data import test_via_path_generator, save_result_with_name
from model import unet

if __name__ == '__main__':
    pretrained_file_name = sys.argv[1]
    data_folder = sys.argv[2]
    result_folder = sys.argv[3]

    model = unet(pretrained_weights=pretrained_file_name)

    sub_data_name = pathlib.PurePath(data_folder).name
    files = os.listdir(data_folder)
    filtered_only_png = list(filter(lambda f: f.endswith('.png'), files))
    sub_data_folder = os.path.join("data", sub_data_name)
    sub_folder = os.path.join(str(pathlib.Path(__file__).parent), sub_data_folder)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    for file in filtered_only_png:
        shutil.copy(os.path.join(data_folder, file), os.path.join(sub_folder, file))

    test_generator = test_via_path_generator(sub_data_folder)
    results = model.predict_generator(test_generator, len(filtered_only_png), verbose=1)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_result_with_name(result_folder, results, filtered_only_png)

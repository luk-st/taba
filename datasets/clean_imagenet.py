import os
from concurrent.futures import ThreadPoolExecutor

import PIL
from PIL import Image
from tqdm import tqdm

DATASET_PATH = "<PATH_TO_IMAGNET>/imagenet64/train"


def process_image(png_file):
    try:
        img = Image.open(os.path.join(DATASET_PATH, png_file))
        del img
    except PIL.UnidentifiedImageError:
        path = os.path.join(DATASET_PATH, png_file)
        print(path)
        os.remove(path)


x = os.listdir(DATASET_PATH)
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, x), total=len(x)))

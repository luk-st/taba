import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import blobfile as bf
import numpy as np
from mpi4py import MPI
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

from . import logger


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    dl_seed=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        data_dir,
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    else:
        generator = torch.Generator().manual_seed(dl_seed) if dl_seed is not None else None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, generator=generator)
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    logger.log("Sorting image files...")
    s_listdir = sorted(bf.listdir(data_dir))
    logger.log("Sorted...")
    for entry in tqdm(s_listdir, "Listing image files"):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.is_imagenet = None
        self.cache_dataset(data_dir, shard, num_shards)

    def cache_dataset(self, data_dir, shard, num_shards):
        if "imagenet" in data_dir.lower():
            print("Running ImageNet!")
            self.is_imagenet = True
            ds_name = "imagenet"
        elif "cifar" in data_dir.lower():
            print("Running Cifar!")
            self.is_imagenet = False
            ds_name = "cifar"
        else:
            raise NotImplementedError(f"Dataset is neither imagenet nor cifar")
        cached_ds_dir = f"datasets/datasets_cached/{ds_name}_train/"
        os.makedirs(cached_ds_dir, exist_ok=True)
        out_path_filename = f"{num_shards}_{shard}.pkl"
        out_path = f"{cached_ds_dir}{out_path_filename}"

        if os.path.exists(out_path):
            logger.log(f"Reading {ds_name} from file: {out_path_filename}...")
            with open(out_path, "rb") as file:
                self.pil_images = pickle.load(file)
            logger.log(f"{ds_name} read from file: {out_path_filename}")
        else:
            self.pil_images = []
            logger.log(f"{ds_name} caching started...")
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.load_image, path): idx for idx, path in enumerate(self.local_images)}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Reading {ds_name} to memory"):
                    idx = futures[future]
                    self.pil_images.insert(idx, future.result())
            logger.log(f"Saving {ds_name} to file: {out_path_filename}")
            with open(out_path, "wb") as file:
                pickle.dump(self.pil_images, file)
            logger.log(f"{ds_name} saved to file: {out_path_filename}")

    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image

    def __getitem__(self, idx):
        pil_image = self.pil_images[idx]
        # pil_image = self.load_image(path=self.local_images[idx])

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX) # type: ignore

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC) # type: ignore

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX) # type: ignore

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC) # type: ignore

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

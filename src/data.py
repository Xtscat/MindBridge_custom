import os
from io import BytesIO

import cairosvg
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from kornia.augmentation.container import AugmentationSequential
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

import utils

img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224, 224), (0.8, 1), p = 0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness = (0.8, 1.2), clip_output = True, p = 0.2),
    kornia.augmentation.RandomContrast(contrast = (0.8, 1.2), clip_output = True, p = 0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p = 0.2),
    kornia.augmentation.RandomSaturation((0.8, 1.2), p = 0.2),
    kornia.augmentation.RandomHue((-0.1, 0.1), p = 0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p = 0.2),
    kornia.augmentation.RandomGrayscale(p = 0.2),
    data_keys = ["input"],
)


class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions = None, pool_num = 8192, pool_type = "max", length = None):
        self.root_dir = root_dir
        self.extensions = extensions if extensions else []
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        self.length = length
        preproc = v2.Compose(
            [
                # v2.Resize(size = self.clip_size[0], interpolation = v2.InterpolationMode.BICUBIC, antialias = None),
                v2.ToDtype(torch.float32, scale = True),
                # v2.CenterCrop(size = self.clip_size),
                v2.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.preprocess = preproc
        if length is not None:
            if length > len(self.samples_keys):
                pass  # enlarge the dataset
            elif length > 0:
                self.samples_keys = self.samples_keys[: length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length :]
            elif length == 0:
                raise ValueError("length must be a non-zero value!")
        else:
            self.length = len(self.samples_keys)

    def _load_samples(self):
        files = os.listdir(self.root_dir)
        samples = {}
        for file in files:
            file_path = os.path.join(self.root_dir, file)
            sample_id, ext = file.split(".", maxsplit = 1)
            if ext in self.extensions:
                if sample_id in samples.keys():
                    samples[sample_id][ext] = file_path
                else:
                    samples[sample_id] = {"subj": file_path}
                    samples[sample_id][ext] = file_path
        return samples

    def _load_image_clip(self, image_path):
        image = read_image(image_path, mode = ImageReadMode.RGB)

        return image

    def _load_image_low_level(self, image_path):
        # image = read_image(image_path, mode = ImageReadMode.RGB)
        # image = self.preprocess(image)
        image = image_path

        return image

    # def _load_image_clip(self, image_path):
    #     image = Image.open(image_path).convert('RGB')
    #     image = np.array(image).astype(np.float32) / 255.0
    #     image = torch.from_numpy(image.transpose(2, 0, 1))
    #
    #     return image

    def _load_image_vae(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_resized = tf.resize(image, 512)
        image_croped = tf.center_crop(image_resized, 512)
        image_data = tf.to_tensor(image_croped)

        return image_data

    def _load_npy(self, npy_path):
        array = np.load(npy_path)
        array = torch.from_numpy(array)
        return array

    def _load_pt(self, pt_path):
        data = torch.load(pt_path)
        return data

    def _load_caption(self, byte_data):
        byte_data = byte_data.numpy()
        return byte_data.tobytes().decode()

    def vox_process(self, x):
        if self.pool_num is not None:
            x = pool_voxels(x, self.pool_num, self.pool_type)
        return x

    def subj_process(self, key):
        id = int(key.split("/")[-2].split("subj")[-1])
        return id

    def aug_process(self, brain3d):
        return brain3d

    def __len__(self):
        # return len(self.samples_keys)
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image_clip(sample[ext]))
            if ext == 'low_level.png':
                items.append(self._load_image_low_level(sample[ext]))
            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                items.append(self.vox_process(voxel))
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))
            elif ext == "subj":
                items.append(self.subj_process(sample[ext]))
            elif ext == "caption.npy":
                byte_data = self._load_npy(sample[ext])
                items.append(self._load_caption(byte_data))
            elif ext == "wholebrain_3d.npy":
                brain3d = self._load_npy(sample[ext])
                items.append(self.aug_process(brain3d))
            elif ext == "emb.pt":
                items.append(self._load_pt(sample[ext]))
            elif ext == "img_rec.pt":
                items.append(self._load_pt(sample[ext]))
        return items


def pool_voxels(voxels, pool_num, pool_type):
    voxels = voxels.float()
    if pool_type == 'avg':
        voxels = nn.AdaptiveAvgPool1d(pool_num)(voxels)
    elif pool_type == 'max':
        voxels = nn.AdaptiveMaxPool1d(pool_num)(voxels)
    elif pool_type == "resize":
        voxels = voxels.unsqueeze(1)  # Add a dimension to make it (B, 1, L)
        voxels = F.interpolate(voxels, size = pool_num, mode = 'linear', align_corners = False)
        voxels = voxels.squeeze(1)

    return voxels


def get_dataloader(
    root_dir,
    batch_size,
    num_workers = 1,
    seed = 42,
    is_shuffle = True,
    extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj", "emb.pt", "img_rec.pt"],
    pool_type = None,
    pool_num = None,
    length = None,
    prefetch_factor = None
):
    utils.seed_everything(seed)
    dataset = NSDDataset(
        root_dir = root_dir, extensions = extensions, pool_num = pool_num, pool_type = pool_type, length = length
    )
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        # pin_memory = True,
        shuffle = is_shuffle,
        prefetch_factor = prefetch_factor
    )

    return dataloader


def get_dls(
    subject = None,
    data_path = None,
    batch_size = None,
    val_batch_size = None,
    extensions = None,
    num_workers = None,
    pool_type = None,
    pool_num = None,
    length = None,
    seed = None
):
    train_path = "{}/webdataset_avg_split/train/subj0{}".format(data_path, subject)
    val_path = "{}/webdataset_avg_split/val/subj0{}".format(data_path, subject)

    # extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj", "emb.pt", "img_rec.pt"]

    # extensions = ['nsdgeneral.npy', "subj", "emb.pt", "img_rec.pt"]  # data for train vae
    # extensions = ['nsdgeneral.npy', 'coco73k.npy', "subj"] # data for train fmri text
    # extensions = ['nsdgeneral.npy', 'jpg', "subj"]  # data for train fmri image sketch or GIT

    train_dl = get_dataloader(
        train_path,
        batch_size = batch_size,
        num_workers = num_workers,
        seed = seed,
        extensions = extensions,
        pool_type = pool_type,
        pool_num = pool_num,
        is_shuffle = True,
        length = length,
    )

    val_dl = get_dataloader(
        val_path,
        batch_size = val_batch_size,
        num_workers = num_workers,
        seed = seed,
        extensions = extensions,
        pool_type = pool_type,
        pool_num = pool_num,
        is_shuffle = False,
    )

    num_train = len(train_dl.dataset)
    num_val = len(val_dl.dataset)
    print(train_path, "\n", val_path)
    print("number of train data:", num_train)
    print("batch_size", batch_size)
    print("number of val data:", num_val)
    print("val_batch_size", val_batch_size)

    return train_dl, val_dl

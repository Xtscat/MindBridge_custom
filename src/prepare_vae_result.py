import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from diffusers.models import AutoencoderKL, AutoencoderTiny
from PIL import Image
from torchvision import transforms


def get_image_paths(folder):
    types = ('*.jpg', )  # 图片文件的扩展名
    image_files = []
    for ext in types:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    return [os.path.abspath(image) for image in image_files]


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1))
    return image


def load_image_vae(image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = tf.resize(image, 512)
    image_croped = tf.center_crop(image_resized, 512)
    image_data = tf.to_tensor(image_croped)

    return image_data


class SDXL_VAE(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super().__init__()

        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype = torch.float16)
        # vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype = torch.float16)
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype = torch.float16)
        vae.requires_grad_(False)
        vae.eval()

        self.latent_channels = vae.config.latent_channels
        self.scaling_factor = vae.config.scaling_factor
        self.vae = vae.to("cuda:0")

    def encode_image(self, images):
        with torch.no_grad():
            embeddings = self.vae.encoder(images)
        return embeddings

    def decode_image(self, embeddings):
        with torch.no_grad():
            images = self.vae.decoder(embeddings).clamp(0, 1)
        return images


vae = SDXL_VAE()

data_root = "/media/SSD_1_2T/xt/data/natural-scenes-dataset/webdataset_avg_split/"
train_root = data_root + "train/"
test_root = data_root + "test/"
val_root = data_root + "val/"
subj_list = ['subj01/', 'subj02/', 'subj05/', 'subj07/']
for subj in subj_list:
    train_root = data_root + "train/"
    test_root = data_root + "test/"
    val_root = data_root + "val/"

    train_root += subj
    test_root += subj
    val_root += subj

    train_img_path = sorted(get_image_paths(train_root))
    test_img_path = sorted(get_image_paths(test_root))
    val_img_path = sorted(get_image_paths(val_root))

    for i in range(0, len(train_img_path), 100):
        batch_img_paths = train_img_path[i : i + 100]
        batch_images = []
        for img_path in batch_img_paths:
            image = load_image_vae(img_path).half().to("cuda:0")
            image = image.unsqueeze(0)
            batch_images.append(image)
        batch_images = torch.cat(batch_images, dim = 0)

        embedding = vae.encode_image(batch_images)
        img_rec = vae.decode_image(embedding)

        for j, img_path in enumerate(batch_img_paths):
            base_path = os.path.splitext(img_path)[0]
            torch.save(embedding[j].cpu(), base_path + '.emb.pt')
            torch.save(img_rec[j].cpu(), base_path + '.img_rec.pt')

        print(f"train_img in {subj}: {i+100} in {len(train_img_path)}")

    for i in range(0, len(test_img_path), 100):
        batch_img_paths = test_img_path[i : i + 100]
        batch_images = []
        for img_path in batch_img_paths:
            image = load_image_vae(img_path).half().to("cuda:0")
            image = image.unsqueeze(0)
            batch_images.append(image)
        batch_images = torch.cat(batch_images, dim = 0)

        embedding = vae.encode_image(batch_images)
        img_rec = vae.decode_image(embedding)

        for j, img_path in enumerate(batch_img_paths):
            base_path = os.path.splitext(img_path)[0]
            torch.save(embedding[j].cpu(), base_path + '.emb.pt')
            torch.save(img_rec[j].cpu(), base_path + '.img_rec.pt')

        print(f"test_img in {subj}: {i+100} in {len(test_img_path)}")

    for i in range(0, len(val_img_path), 100):
        batch_img_paths = val_img_path[i : i + 100]
        batch_images = []
        for img_path in batch_img_paths:
            image = load_image_vae(img_path).half().to("cuda:0")
            image = image.unsqueeze(0)
            batch_images.append(image)
        batch_images = torch.cat(batch_images, dim = 0)

        embedding = vae.encode_image(batch_images)
        img_rec = vae.decode_image(embedding)

        for j, img_path in enumerate(batch_img_paths):
            base_path = os.path.splitext(img_path)[0]
            torch.save(embedding[j].cpu(), base_path + '.emb.pt')
            torch.save(img_rec[j].cpu(), base_path + '.img_rec.pt')

        print(f"val_img in {subj}: {i+100} in {len(val_img_path)}")

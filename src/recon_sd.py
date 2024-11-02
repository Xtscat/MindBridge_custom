import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2
from transformers import (CLIPTextModelWithProjection, CLIPTokenizer,
                          CLIPVisionModelWithProjection)


def seed_everything(seed = 0, cudnn_deterministic = True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')


def prepare_SD(device):
    model_id = "stabilityai/stable-diffusion-2-base"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype = torch.float16, cache_dir = "/media/SSD_1_2T/xt/weights/"
    )
    pipe = pipe.to(device)

    return pipe


def prepare_prompt():
    """ must return a string """
    pass


def prepare_lowlevel_img():
    """ must return a PIL.Image """
    pass


def prepare_coco_img():
    """ must return a PIL.Image """


def prepare_clip(device):
    """ return clip mocel """
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir = "/media/SSD_1_2T/xt/weights/",
    ).eval()
    image_encoder = image_encoder.to(device)
    for param in image_encoder.parameters():
        param.requires_grad = False

    # text_encoder = CLIPTextModelWithProjection.from_pretrained(
    #     "openai/clip-vit-large-patch14", cache_dir = "/media/SSD_1_2T/xt/weights/"
    # ).eval()
    # text_encoder = text_encoder.to(device)
    # for param in text_encoder.parameters():
    #     param.requires_grad = False
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return image_encoder


def embed_image(image_encoder, image):
    def versatile_normalize_embeddings(encoder_output):
        embeds = encoder_output.last_hidden_state
        embeds = image_encoder.vision_model.post_layernorm(embeds)
        embeds = image_encoder.visual_projection(embeds)
        return embeds

    preprocess = v2.Compose(
        [
            v2.Resize(size = 224, interpolation = v2.InterpolationMode.BICUBIC, antialias = None),
            v2.ToDtype(torch.float32, scale = True),
            v2.CenterCrop(size = 224),
            v2.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711))
        ]
    )

    clip_emb = preprocess(transforms.ToTensor()(image).to(image_encoder.device))
    clip_emb = image_encoder(clip_emb.unsqueeze(0))
    clip_emb = versatile_normalize_embeddings(clip_emb)
    return clip_emb


def pick_image(image_encoder, coco_img, lowlevel_img = None, images: list = None):
    if lowlevel_img is not None:
        plot_start_idx = 2
    else:
        plot_start_idx = 1

    best_picks = np.zeros(len(images)).astype(np.int16)

    clip_image_target = embed_image(image_encoder, coco_img)
    clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim = -1)
    sims = []

    for im in images:
        currecon = embed_image(image_encoder, im)
        currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim = -1)
        cos_sim = nn.functional.cosine_similarity(clip_image_target_norm, currecon)
        sims.append(cos_sim.item())
    best_picks[0] = int(np.nanargmax(sims))
    print(sims)
    print(best_picks[0])

    num_images = len(images) + plot_start_idx

    fig, axes = plt.subplots(1, num_images, figsize = ((1 + len(images)) * 5, 6), facecolor = (1, 1, 1))
    axes = axes.flatten()

    if lowlevel_img is not None:
        axes[0].imshow(coco_img)
        axes[0].axis('off')
        axes[0].set_title('original_img')
        axes[1].imshow(lowlevel_img)
        axes[1].axis('off')
        axes[1].set_title('lowlevel_img')
    else:
        axes[0].imshow(coco_img)
        axes[0].axis('off')
        axes[0].set_title('original_img')

    for i in range(plot_start_idx, num_images):
        if (i - plot_start_idx) == best_picks[0]:
            axes[i].set_title('best_rec')
        axes[i].imshow(images[i - plot_start_idx])
        axes[i].axis('off')

    return fig, best_picks[0]


def main(device):
    seed_everything(42)
    image_encoder = prepare_clip(device)
    sd_pipe = prepare_SD(device)
    prompt = prepare_prompt()
    # coco_img = prepare_coco_img()
    # lowlevel_img = prepare_lowlevel_img()
    coco_img = Image.open(
        "/media/SSD_1_2T/xt/data/natural-scenes-dataset/webdataset_avg_split/test/subj01/sample000000383.jpg"
    )
    lowlevel_img = Image.open("/media/SSD_1_2T/xt/MindBridge/src/test_lowlevel.png")
    prompt = "a real world picture: a black bear sitting on the road with the blue sky as background"
    negative_prompt = "cartoon, blurry, artificial, ugly, paintings, anime, monochrome, worst quality, low quality, normal quality, lowres"

    # images: a list of PIL.Image.Image
    images = sd_pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        image = lowlevel_img,
        strength = 0.8,
        num_inference_steps = 80,
        guidance_scale = 12,
        num_images_per_prompt = 8
    ).images

    fig, best_pick = pick_image(
        image_encoder = image_encoder, coco_img = coco_img, lowlevel_img = lowlevel_img, images = images
    )
    picked_img = images[best_pick.astype(np.int8)]

    fig.savefig("test.png")
    picked_img.save("test_best.png")


if __name__ == "__main__":
    device = "cuda:1"
    main(device)

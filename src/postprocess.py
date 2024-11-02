import io
import os
import re

import cairosvg
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from models import MindSingle_text
from options import args


def prepare_sdct(args, device):
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype = torch.float16, cache_dir = "/media/SSD_1_2T/xt/weights/"
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        controlnet = controlnet,
        torch_dtype = torch.float16,
        cache_dir = "/media/SSD_1_2T/xt/weights/"
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.to(device)

    generator = torch.manual_seed(42)
    return pipe, generator


def prepare_sdxlct(args, device):
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionXLControlNetImg2ImgPipeline)
    from transformers import DPTImageProcessor

    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype = torch.float16, cache_dir = "/media/SSD_1_2T/xt/weights/"
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype = torch.float16, cache_dir = "/media/SSD_1_2T/xt/weights/"
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet = controlnet,
        vae = vae,
        variant = "fp16",
        use_safetensors = True,
        torch_dtype = torch.float16,
    ).to(device)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(42)
    return pipe, generator


def prepare_GIT(device):
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/git-large-coco", cache_dir = "/media/SSD_1_2T/xt/weights/"
    ).to(device)
    return processor, model


def generate_captions(processor, model, img):
    pixel_values = processor(images = img, return_tensors = "pt").pixel_values.to(model.device)
    generated_ids = model.generate(pixel_values = pixel_values, max_length = 50)
    generate_captions = processor.batch_decode(generated_ids, skip_special_tokens = True)[0]
    return generate_captions


def sort_keys(s):
    return int(s.split('/')[-1].split('_')[0])


def prepare_img(img_folder_path):
    all_files = [
        os.path.join(img_folder_path, f)
        for f in os.listdir(img_folder_path)
        if os.path.isfile(os.path.join(img_folder_path, f))
    ]
    rec_pt_files = [f for f in all_files if f.endswith('rec.pt')]
    rec_pt_files = sorted(rec_pt_files, key = sort_keys)

    img_pt_files = [f for f in all_files if f.endswith('img.pt')]
    img_pt_files = sorted(img_pt_files, key = sort_keys)
    return rec_pt_files, img_pt_files


def prepare_sketch_and_image(sketch_folder_path):
    all_subjs = [os.path.join(sketch_folder_path, folder) for folder in os.listdir(sketch_folder_path)]
    all_subjs = sorted(all_subjs)
    all_subjs_name = []
    all_svg, all_coco, all_vd = [], [], []
    for i, subj in enumerate(all_subjs):
        if os.path.exists(os.path.join(subj, "runs")):
            subj = os.path.join(subj, "runs")
        else:
            continue

        svg_name = os.listdir(subj)[0] + "_seed42_best.svg"
        png_folder_name = os.listdir(subj)[0] + "_seed42"

        subj = os.path.join(subj, os.listdir(subj)[0])
        subj_folder = os.path.join(subj, "_fmri")
        subj_png_folder = os.path.join(subj_folder, png_folder_name)

        if os.path.exists(os.path.join(subj_folder, svg_name)):
            subj_svg = os.path.join(subj_folder, svg_name)
        else:
            continue
        if os.path.exists(os.path.join(subj_png_folder, "coco_img.png")):
            subj_coco = os.path.join(subj_png_folder, "coco_img.png")
        else:
            continue
        if os.path.exists(os.path.join(subj_png_folder, "vd_img.png")):
            subj_vd = os.path.join(subj_png_folder, "vd_img.png")
        else:
            continue

        all_svg.append(subj_svg)
        all_coco.append(subj_coco)
        all_vd.append(subj_vd)
        all_subjs_name.append(subj_svg.split('/')[6])

    return all_svg, all_coco, all_vd, all_subjs_name


def prepare_fmri(fmri_folder_path, subj_name, device):
    fmri_path = os.path.join(fmri_folder_path, subj_name + '.nsdgeneral.npy')
    fmri = np.load(fmri_path)
    fmri = torch.from_numpy(fmri).float().to(device)
    fmri = nn.AdaptiveAvgPool1d(8192)(fmri)
    fmri = torch.mean(fmri, dim = 0)
    return fmri


def prepare_voxel2clip():
    voxel2clip_kwargs = dict(in_dim = 8192, out_dim_text = 77 * 768, h = 2048, n_blocks = 4, subj_list = [1, ])

    voxel2clip = MindSingle_text(**voxel2clip_kwargs)

    outdir = '/media/SSD_1_2T/xt/MindBridge/train_logs/MindBrige_text_infonce_multicoco_for_duffusion/'
    ckpt_path = os.path.join(outdir, 'last.pth')
    print("ckpt_path", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location = 'cpu')
    print("EPOCH: ", checkpoint['epoch'])
    state_dict = checkpoint['model_state_dict']

    voxel2clip.load_state_dict(state_dict, strict = False)
    voxel2clip.requires_grad_(False)
    voxel2clip.eval().to(device)

    return voxel2clip


def main(
    postprocess_folder: str = None,
    caption_path: str = None,
    img_folder_path: str = None,
    sketch_folder_path: str = None,
    fmri_folder_path: str = None,
    device: str = None
):
    # sdct_pipe, generator = prepare_sdxlct(args, device)
    sdct_pipe, generator = prepare_sdct(args, device)
    git_processor, git_model = prepare_GIT(device)
    # voxel2clip = prepare_voxel2clip()

    outdir = f'../train_logs/{postprocess_folder}_svg'
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok = True)

    sketches, coco_imgs, rec_imgs, subj_names = prepare_sketch_and_image(sketch_folder_path)
    # captions = pd.read_excel('/media/SSD_1_2T/xt/MindBridge/src/captions.xlsx')

    assert isinstance(rec_imgs, list) and isinstance(coco_imgs, list) and isinstance(
        sketches, list
    ), "The type of captions and images and sketchs must be list"
    assert len(rec_imgs) and len(coco_imgs) == len(
        sketches
    ), f"The length of captions and images and sketchs must same. rec_imgs: {len(rec_imgs)}, coco_imgs: {len(coco_imgs)}, sketchs: {len(sketches)}"

    for rec_img, coco_img, sketch, subj_name in tqdm(
        zip(rec_imgs, coco_imgs, sketches, subj_names), total = len(sketches)
    ):
        # subj_id = int(re.search(r'\d+', sketch.split('/')[6]).group())
        # if subj_id <= 3552:
        #     continue
        # print(sketch.split('/')[6])
        print(subj_name)

        save_dir = os.path.join(
            "/media/SSD_1_2T/xt/MindBridge/train_logs/Postprocess_strength0.2_guidance5",
            rec_img.split('/')[6]
        )
        save_dir = os.path.join(save_dir, sketch.split('/')[6])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load img:
        #     [3, 512, 512]
        # print(f"rec_img: {rec_img}")
        # print(f"coco_img: {coco_img}")
        rec_img = load_image(rec_img)
        coco_img = load_image(coco_img)

        vd_img_to_save = rec_img
        vd_img_save_path = os.path.join(save_dir, "vd_img.png")
        vd_img_to_save.save(vd_img_save_path)

        coco_img_to_save = coco_img
        coco_img_save_path = os.path.join(save_dir, "coco_img.png")
        coco_img_to_save.save(coco_img_save_path)

        # load svg and transfrom it to png with write background:
        #     (224, 224)
        sketch = cairosvg.svg2png(url = sketch, background_color = 'white', output_width = 512, output_height = 512)
        sketch = Image.open(io.BytesIO(sketch))
        sketch_save_path = os.path.join(save_dir, "sketch_img.png")
        sketch.save(sketch_save_path)

        # caption
        caption = generate_captions(git_processor, git_model, rec_img)
        print(caption)
        caption_save_path = os.path.join(save_dir, "caption.txt")
        with open(caption_save_path, "w", encoding = "utf-8") as f:
            f.write(caption)
        # fmri = prepare_fmri(fmri_folder_path, subj_name, device)
        # text_embedding = voxel2clip(fmri).reshape(1, -1, 768)

        # negative prompt
        negative_prompt = "sketches, monochrome, grayscale, blurry"

        # postprocess
        with torch.no_grad():
            rec_imgs = sdct_pipe(
                image = rec_img,
                control_image = sketch,
                strength = 0.2,
                num_inference_steps = 100,
                guidance_scale = 5,
                num_images_per_prompt = 1,
                generator = generator,
                # prompt_embeds = text_embedding,
                prompt = caption,
                negative_prompt = negative_prompt
            ).images[0]
            rec_imgs.save(f"postprocessed_img.png")
        postprocess_img_path = os.path.join(save_dir, "postprocessed_img.png")
        rec_imgs.save(postprocess_img_path)


if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    postprocess_folder = "/media/SSD_1_2T/xt/train_logs/SDCT"
    caption_path = "/media/SSD_1_2T/xt/MindBridge/captions.xlsx"
    img_folder_path = "/media/SSD_1_2T/xt/MindBridge/train_logs/VD_text_img_infonce_guidance5_ratio0.5/recon_on_subj1"
    sketch_folder_path = "/media/SSD_1_2T/xt/SceneSketch/results_sketches/"
    fmri_folder_path = "/media/SSD_1_2T/xt/data/natural-scenes-dataset/webdataset_avg_split/test/subj01/"

    main(
        postprocess_folder = postprocess_folder,
        img_folder_path = img_folder_path,
        caption_path = caption_path,
        sketch_folder_path = sketch_folder_path,
        fmri_folder_path = fmri_folder_path,
        device = device
    )

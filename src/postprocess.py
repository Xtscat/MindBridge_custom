import io
import os

import cairosvg
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from options import args


def prepare_sdct(args, device):
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-sd21-scribble-diffusers",
        torch_dtype = torch.float16,
        cache_dir = "/media/SSD_1_2T/xt/weights/"
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet = controlnet,
        torch_dtype = torch.float16,
        cache_dir = "/media/SSD_1_2T/xt/weights/"
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.to(device)

    generator = torch.manual_seed(42)
    return pipe, generator


def sort_keys(s):
    return int(s.split('/')[-1].split('_')[0])


def prepare_captions(caption_path):
    df = pd.read_excel(caption_path)
    captions_from_brain = df['captions_from_brain'].tolist()
    return captions_from_brain


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


def prepare_sketch(sketch_folder_path):
    all_subjs = [os.path.join(sketch_folder_path, folder) for folder in os.listdir(sketch_folder_path)]
    all_subjs = sorted(all_subjs)
    for i, subj in enumerate(all_subjs):
        subj = os.path.join(subj, "runs")
        subj = os.path.join(subj, os.listdir(subj)[0])
        if os.path.exists(os.path.join(subj, subj.split('/')[-1] + "_seed42_best.svg")):
            subj = os.path.join(subj, subj.split('/')[-1] + "_seed42_best.svg")
        elif os.path.exists(os.path.join(subj, subj.split('/')[-1] + "_seed42/svg_logs")):
            subj = os.path.join(subj, subj.split('/')[-1] + "_seed42/svg_logs")
            subj = os.path.join(subj, sorted(os.listdir(subj))[-1])
        else:
            raise ValueError(f"{subj}_svg does not exists")
        all_subjs[i] = subj
    return all_subjs


def main(
    postprocess_folder: str = None,
    caption_path: str = None,
    img_folder_path: str = None,
    sketch_folder_path: str = None,
    device: str = None
):
    sdct_pipe, generator = prepare_sdct(args, device)
    outdir = f'../train_logs/{postprocess_folder}_svg'
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok = True)

    captions = prepare_captions(caption_path)
    rec_imgs, coco_imgs = prepare_img(img_folder_path)
    sketches = prepare_sketch(sketch_folder_path)

    assert isinstance(captions, list) and isinstance(rec_imgs, list) and isinstance(coco_imgs, list) and isinstance(
        sketches, list
    ), "The type of captions and images and sketchs must be list"
    assert len(captions) == len(rec_imgs) and len(captions) == len(coco_imgs) and len(captions) == len(
        sketches
    ), "The length of captions and images and sketchs must same"

    for caption, rec_img, coco_img, sketch in zip(captions, rec_imgs, coco_imgs, sketches):
        """ 
            caption: a man cooking food in a kitchen.
            img: /media/SSD_1_2T/xt/MindBridge/train_logs/VD_text_img_infonce_guidance5_ratio0.5/recon_on_subj1/0_rec.pt
            sketch: /media/SSD_1_2T/xt/data/natural-scenes-dataset/results_sketches_subj1/sample000000349/runs/background_l2_sample000000349/background_l2_sample000000349_seed42_best.svg
        """
        print(sketch.split('/')[7])

        save_dir = os.path.join("/media/SSD_1_2T/xt/MindBridge/train_logs/Postprocess", rec_img.split('/')[7])
        save_dir = os.path.join(save_dir, sketch.split('/')[7])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load img:
        #     [3, 512, 512]
        rec_img = torch.load(rec_img).squeeze(0).squeeze(0)
        coco_img = torch.load(coco_img).squeeze(0).squeeze(0)

        transform = transforms.ToPILImage()

        vd_img_to_save = transform(rec_img)
        vd_img_save_path = os.path.join(save_dir, "vd_img.png")
        vd_img_to_save.save(vd_img_save_path)

        coco_img_to_save = transform(coco_img)
        coco_img_save_path = os.path.join(save_dir, "coco_img.png")
        coco_img_to_save.save(coco_img_save_path)

        # load svg and transfrom it to png with write background:
        #     (224, 224)
        sketch = cairosvg.svg2png(url = sketch, background_color = 'white', output_width = 512, output_height = 512)
        sketch = Image.open(io.BytesIO(sketch))
        sketch_save_path = os.path.join(save_dir, "sketch_img.png")
        sketch.save(sketch_save_path)

        # caption
        print(caption)
        caption = "a photo of natural scenes that: " + caption
        caption_save_path = os.path.join(save_dir, "caption.txt")
        with open(caption_save_path, "w", encoding = "utf-8") as f:
            f.write(caption)

        # postprocess
        with torch.no_grad():
            rec_imgs = sdct_pipe(
                image = rec_img,
                control_image = sketch,
                strength = 3,
                num_inference_steps = 50,
                guidance_scale = 12,
                num_images_per_prompt = 1,
                generator = generator,
                prompt = caption,
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
    sketch_folder_path = "/media/SSD_1_2T/xt/data/natural-scenes-dataset/results_sketches_subj1"

    main(
        postprocess_folder = postprocess_folder,
        img_folder_path = img_folder_path,
        caption_path = caption_path,
        sketch_folder_path = sketch_folder_path,
        device = device
    )

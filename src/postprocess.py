import os
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.utils import load_image
from torchvision.io import write_png
from tqdm import tqdm

import data
import utils
from models import Clipper, MindBridge, MindSingle, Voxel2vae
from nsd_access import NSDAccess
from options import args


def prepare_data(args):
    ## Load data
    subj_num_voxels = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    args.num_voxels = subj_num_voxels[args.subj_test]

    test_path = "{}/webdataset_avg_split/test/subj0{}".format(args.data_path, args.subj_test)
    test_dl = data.get_dataloader(
        test_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        seed = args.seed,
        is_shuffle = False,
        extensions = ['nsdgeneral.npy', "jpg", 'low_level.png', "subj", "caption.npy"],
        pool_type = args.pool_type,
        pool_num = args.pool_num,
    )

    return test_dl


def prepare_sdct(args, device):
    # !pip install opencv-python transformers accelerate
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionControlNetPipeline,
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
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-1",
    #     controlnet = controlnet,
    #     torch_dtype = torch.float16,
    #     cache_dir = "/media/SSD_1_2T/xt/weights/"
    # )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
    pipe.to(device)

    generator = torch.manual_seed(0)
    return pipe, generator


def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    test_dl = prepare_data(args)
    num_test = len(test_dl)

    sdct_pipe, generator = prepare_sdct(args, device)
    outdir = f'../train_logs/{args.model_name}_svg'
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok = True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    test_range = np.arange(num_test)

    for val_i, (voxel, img, img_lowlevel, subj, caption) in enumerate(tqdm(test_dl, total = len(test_range))):
        if val_i <= 1:
            continue
        if val_i > 2:
            break

        img_vd_path = f"/media/SSD_1_2T/xt/MindBridge/train_logs/MindBrige_text_only_mixco_loss/recon_on_subj1/{val_i}_rec.pt"
        print(img_vd_path)
        print(caption)
        img_vd = torch.load(img_vd_path).squeeze(0)

        to_pil = transforms.ToPILImage()
        img_vd_pil = to_pil(img_vd.squeeze(0))
        img_vd_pil.save(f"img_vd.png")

        img_lowlevel = load_image(img_lowlevel[0])
        img_lowlevel.save(f"img_lowlevel.png")

        voxel = torch.mean(voxel, axis = 1).float().to(device)
        write_png(img.squeeze(0), "img_original.png")
        img = img.to(device)

        with torch.no_grad():
            images = sdct_pipe(
                image = img_vd,
                control_image = img_lowlevel,
                strength = 1.0,
                num_inference_steps = 50,
                guidance_scale = 12,
                num_images_per_prompt = 1,
                generator = generator,
                prompt = caption[0],
            ).images[0]
            images.save(f"postprocessed_image.png")

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("save path:", save_dir)


if __name__ == "__main__":
    utils.seed_everything(seed = args.seed)
    args.model_name = "MindBrige_text_only_mixco_loss_diffusers_vit_norm"
    args.ckpt_from = "last"
    args.h_size = 2048
    args.n_blocks = 4
    args.pool_type = "max"
    args.subj_load = [1]
    args.subj_test = 1
    args.pool_num = 8192

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    main(device)

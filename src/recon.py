import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import data
import utils
from eval import cal_metrics
from models import (Clipper, MindBridge_image, MindBridge_text,
                    MindSingle_image, MindSingle_text, Voxel2vae)
from nsd_access import NSDAccess
from options import args


## Load autoencoder
def prepare_voxel2sd(args, ckpt_path, device):
    from models import Voxel2StableDiffusionModel
    checkpoint = torch.load(ckpt_path, map_location = device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim = args.num_voxels)

    voxel2sd.load_state_dict(state_dict, strict = False)
    voxel2sd.to(device)
    voxel2sd.eval()
    print("Loaded low-level model!")

    return voxel2sd


def prepare_coco(args):
    # Preload coco captions
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k, info_type = 'captions')

    print("coco captions loaded.")

    return prompts_list


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
        extensions = ['nsdgeneral.npy', "jpg", 'low_level.png', 'coco73k.npy', "subj"],
        pool_type = args.pool_type,
        pool_num = args.pool_num,
    )

    return test_dl


def prepare_VD(args, device):
    print('Creating versatile diffusion reconstruction pipeline...')
    from diffusers import (UniPCMultistepScheduler,
                           VersatileDiffusionDualGuidedPipeline)
    from diffusers.models import DualTransformer2DModel
    try:
        vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("/media/SSD_1_2T/xt/weights/")
    except:
        print("Downloading Versatile Diffusion to", args.diffusion_cache_dir)
        vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion", cache_dir = "/media/SSD_1_2T/xt/weights/"
        )

    vd_pipe.image_unet.eval().to(device)
    vd_pipe.vae.eval().to(device)
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)

    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
        "shi-labs/versatile-diffusion", cache_dir = "/media/SSD_1_2T/xt/weights/", subfolder = "scheduler"
    )

    # Set weighting of Dual-Guidance
    # text_image_ratio=0.5 means equally weight text and image, 0 means use only image
    for name, module in vd_pipe.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = args.text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    return vd_pipe


def prepare_SD(args, device):
    print('Creating controlnet stable diffusion reconstruction pipeline...')
    from diffusers import StableDiffusionPipeline

    from inference_pipe import fMRI2ImgDiffusionPipeline
    try:
        # sd_pipe = fMRI2ImgDiffusionPipeline.from_pretrained(args.diffusion_cache_dir)
        sd_pipe = StableDiffusionPipeline.from_pretrained(args.diffusion_cache_dir)
    except:
        print("Downloading controlnet stable diffusion to", args.diffusion_cache_dir)
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype = torch.float32,
            variant = "fp16",
            cache_dir = args.diffusion_cache_dir
        )
        # sd_pipe = fMRI2ImgDiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-1",
        #     torch_dtype = torch.float32,
        #     variant = "fp16",
        #     cache_dir = args.diffusion_cache_dir
        # )

    # sd_pipe.unet.eval().to(device)
    # sd_pipe.vae.eval().to(device)
    # sd_pipe.text_encoder.eval().to(device)
    # sd_pipe.unet.requires_grad_(False)
    # sd_pipe.vae.requires_grad_(False)
    # sd_pipe.text_encoder.requires_grad_(False)
    # sd_pipe.eval().to(device)
    # sd_pipe.requires_grad_(False)
    sd_pipe = sd_pipe.to(device)

    return sd_pipe


def prepare_SD(args, device):
    print('Creating controlnet stable diffusion reconstruction pipeline...')
    from diffusers import StableDiffusionPipeline

    from inference_pipe import fMRI2ImgDiffusionPipeline
    try:
        # sd_pipe = fMRI2ImgDiffusionPipeline.from_pretrained(args.diffusion_cache_dir)
        sd_pipe = StableDiffusionPipeline.from_pretrained(args.diffusion_cache_dir)
    except:
        print("Downloading controlnet stable diffusion to", args.diffusion_cache_dir)
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype = torch.float32,
            variant = "fp16",
            cache_dir = args.diffusion_cache_dir
        )
    sd_pipe = sd_pipe.to(device)

    return sd_pipe


def prepare_CLIP(args, device):
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes[args.clip_variant]
    out_dim_image = 257 * clip_size
    out_dim_text = 77 * clip_size
    clip_extractor = Clipper("ViT-L/14", hidden_state = True, norm_embs = True, device = device)

    return clip_extractor


def prepare_voxel2clip(args, out_dim_image, out_dim_text, device, mode):
    assert mode in ["text", "image"], "mode must in ['text','image']"

    if mode == "text":
        voxel2clip_kwargs = dict(
            in_dim = args.pool_num,
            out_dim_text = out_dim_text,
            h = args.h_size,
            n_blocks = args.n_blocks,
            subj_list = args.subj_load
        )

        # only need to load Single-subject version of MindBridge
        voxel2clip = MindSingle_text(**voxel2clip_kwargs)

        outdir = "/media/SSD_1_2T/xt/MindBridge/train_logs/MindBrige_text_infonce/"
        # outdir = f'../train_logs/{args.model_name}'
        ckpt_path = os.path.join(outdir, f'{args.ckpt_from}.pth')
        print("ckpt_path", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location = 'cpu')
        print("EPOCH: ", checkpoint['epoch'])
        state_dict = checkpoint['model_state_dict']

        voxel2clip.load_state_dict(state_dict, strict = False)
        voxel2clip.requires_grad_(False)
        voxel2clip.eval().to(device)

    if mode == "image":
        voxel2clip_kwargs = dict(
            in_dim = args.pool_num,
            out_dim_image = out_dim_image,
            h = args.h_size,
            n_blocks = args.n_blocks,
            subj_list = args.subj_load
        )

        # only need to load Single-subject version of MindBridge
        voxel2clip = MindSingle_image(**voxel2clip_kwargs)

        outdir = "/media/SSD_1_2T/xt/MindBridge/train_logs/MindBrige_img_infonce/"
        # outdir = f'../train_logs/{args.model_name}'
        ckpt_path = os.path.join(outdir, f'{args.ckpt_from}.pth')
        print("ckpt_path", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location = 'cpu')
        print("EPOCH: ", checkpoint['epoch'])
        state_dict = checkpoint['model_state_dict']

        voxel2clip.load_state_dict(state_dict, strict = False)
        voxel2clip.requires_grad_(False)
        voxel2clip.eval().to(device)

    return voxel2clip


def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    # Load data
    test_dl = prepare_data(args)
    num_test = len(test_dl)

    # Load autoencoder
    outdir_ae = f'../train_logs/{args.autoencoder_name}'
    ckpt_path = os.path.join(outdir_ae, f'epoch120.pth')
    if os.path.exists(ckpt_path):
        voxel2sd = prepare_voxel2sd(args, ckpt_path, device)
        # pool later
        args.pool_type = None
    else:
        print("No valid path for low-level model specified; not using img2img!")
        args.img2img_strength = 1

    # # Load VD pipeline
    vd_pipe = prepare_VD(args, device)
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    # Load CLIP
    clip_extractor = prepare_CLIP(args, device)

    # load voxel2clip
    voxel2clip_text = prepare_voxel2clip(args, None, 77 * 768, device, mode = "text")
    voxel2clip_image = prepare_voxel2clip(args, 257 * 768,None, device, mode = "image")

    outdir = f'../train_logs/VD_text_img_infonce_guidance5_ratio0.5/'
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok = True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # define test range
    test_range = np.arange(num_test)
    if args.test_end is None:
        args.test_end = num_test

    # define recon logic
    only_lowlevel = False
    if args.img2img_strength == 1:
        img2img = False
    elif args.img2img_strength == 0:
        img2img = True
        only_lowlevel = True
    else:
        img2img = True

    # recon loop
    for val_i, (voxel, img, img_lowlevel, coco, subj) in enumerate(tqdm(test_dl, total = len(test_range))):
        if val_i < args.test_start:
            continue
        if val_i >= args.test_end:
            break
        if (args.samples is not None) and (val_i not in args.samples):
            continue

        voxel = torch.mean(voxel, axis = 1).float().to(device)
        img = img.to(device)

        with torch.no_grad():
            if args.only_embeddings:
                results = voxel2clip_text(voxel)
                embeddings = results[: 2]
                torch.save(embeddings, os.path.join(save_dir, f'embeddings_{val_i}.pt'))
                continue
            if img2img:  # will apply low-level and high-level pipeline
                ae_preds = voxel2sd(voxel)
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device) / 0.18215).sample / 2 + 0.5

                if val_i == 0:
                    plt.imshow(utils.torch_to_Image(blurry_recons))
                    plt.show()

                # pooling
                voxel = data.pool_voxels(voxel, args.pool_num, args.pool_type)
            else:  # only high-level pipeline
                blurry_recons = None

            if only_lowlevel:  # only low-level pipeline
                brain_recons = blurry_recons
            else:
                grid, brain_recons, best_picks, recon_img = utils.reconstruction(
                    img,
                    voxel,
                    voxel2clip_text,
                    voxel2clip_image,
                    clip_extractor,
                    unet,
                    vae,
                    noise_scheduler,
                    img_lowlevel = blurry_recons,
                    num_inference_steps = args.num_inference_steps,
                    n_samples_save = args.batch_size,
                    recons_per_sample = args.recons_per_sample,
                    guidance_scale = args.guidance_scale,
                    img2img_strength = args.img2img_strength,  # 0=fully rely on img_lowlevel, 1=not doing img2img
                    seed = args.seed,
                    plotting = args.plotting,
                    verbose = args.verbose,
                    device = device,
                    mem_efficient = False,
                )

                if args.plotting:
                    grid.savefig(os.path.join(save_dir, f'{val_i}.png'))

                brain_recons = brain_recons[:, best_picks.astype(np.int8)]

                torch.save(img, os.path.join(save_dir, f'{val_i}_img.pt'))
                torch.save(brain_recons, os.path.join(save_dir, f'{val_i}_rec.pt'))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("save path:", save_dir)


if __name__ == "__main__":
    utils.seed_everything(seed = args.seed)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    main(device)

    # args.results_path = f'../train_logs/{args.model_name}/recon_on_subj{args.subj_test}'
    # cal_metrics(args.results_path, device)

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

import data
import utils
from models import Clipper
from nsd_access import NSDAccess
from options import args


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
        extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
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
        "shi-labs/versatile-diffusion", cache_dir = "/media/SSD_1_2T/xt/MindBridge/weights/", subfolder = "scheduler"
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


def prepare_CLIP(args, device):
    clip_extractor = Clipper("ViT-L/14", hidden_state = True, norm_embs = True, device = device)

    return clip_extractor


def decode_latents(latents, vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image/2 + 0.5).clamp(0, 1)
    return image


def batchwise_cosine_similarity(Z, B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim = 1, keepdim = True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim = 0, keepdim = True)  # Size (1, b).
    cosine_similarity = ((Z@B) / (Z_norm@B_norm)).T
    return cosine_similarity


def torch_to_Image(x):
    if x.ndim == 4:
        x = x[0]
    return transforms.ToPILImage()(x)


def reconstruction(
    image,
    text_embedding,
    image_embedding,
    clip_extractor,
    unet,
    vae,
    noise_scheduler,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    seed = 42,
    plotting = True,
    verbose = False,
    n_samples_save = 1,
    device = None,
    mem_efficient = True,
):
    assert n_samples_save == 1, "n_samples_save must = 1. Function must be called one image at a time"
    assert recons_per_sample > 0, "recons_per_sample must > 0"

    brain_recons = None

    if verbose: print("input_image", image.shape, image.dtype)
    B = 1

    if mem_efficient:
        clip_extractor.to("cpu")
        unet.to("cpu")
        vae.to("cpu")
    else:
        clip_extractor.to(device)
        unet.to(device)
        vae.to(device)

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device = device)
    generator.manual_seed(seed)

    brain_clip_text_embeddings = text_embedding
    brain_clip_image_embeddings = image_embedding

    brain_clip_image_embeddings = brain_clip_image_embeddings.repeat(recons_per_sample, 1, 1)
    brain_clip_text_embeddings = brain_clip_text_embeddings.repeat(recons_per_sample, 1, 1)

    if recons_per_sample > 0:
        for samp in range(len(brain_clip_text_embeddings)):
            brain_clip_image_embeddings[samp] = brain_clip_image_embeddings[samp] / (
                brain_clip_image_embeddings[samp, 0].norm(dim = -1).reshape(-1, 1, 1) + 1e-6
            )
            brain_clip_text_embeddings[samp] = brain_clip_text_embeddings[samp] / (
                brain_clip_text_embeddings[samp, 0].norm(dim = -1).reshape(-1, 1, 1) + 1e-6
            )
        input_embedding = brain_clip_image_embeddings  #.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding", input_embedding.shape)

        prompt_embeds = brain_clip_text_embeddings
        if verbose: print("prompt_embedding", prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # 3. dual_prompt_embeddings
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim = 1)
        # input_embedding = prompt_embeds

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps = num_inference_steps, device = device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2  # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None:  # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start :]
            latent_timestep = timesteps[: 1].repeat(batch_size)

            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            if mem_efficient:
                vae.to(device)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn(
                [recons_per_sample, 4, 64, 64], device = device, generator = generator, dtype = input_embedding.dtype
            )
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn(
                [recons_per_sample, 4, 64, 64], device = device, generator = generator, dtype = input_embedding.dtype
            )
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        if mem_efficient:
            unet.to(device)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            if verbose:
                print(
                    "timesteps: {}, latent_model_input: {}, input_embedding: {}".format(
                        i, latent_model_input.shape, input_embedding.shape
                    )
                )
            noise_pred = unet(latent_model_input, t, encoder_hidden_states = input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text-noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        if mem_efficient:
            unet.to("cpu")

        recons = decode_latents(latents.to(device), vae.to(device)).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons", brain_recons.shape, brain_recons.dtype)

    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)

    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)

    clip_image_target = clip_extractor.embed_image(image)
    clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim = -1)
    sims = []
    for im in range(recons_per_sample):
        # compute cosine sims
        currecon = clip_extractor.embed_image(brain_recons[0, [im]].float()).to(clip_image_target_norm.device
                                                                                ).to(clip_image_target_norm.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim = -1)
        cursim = batchwise_cosine_similarity(clip_image_target_norm, currecon)
        sims.append(cursim.item())

    if verbose: print(sims)
    best_picks[0] = int(np.nanargmax(sims))
    if verbose: print(best_picks)
    if mem_efficient:
        clip_extractor.to("cpu")

    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1 + img2img_samples + recons_per_sample
    if plotting:
        fig, ax = plt.subplots(
            n_samples_save,
            num_xaxis_subplots,
            figsize = (num_xaxis_subplots * 5, 6 * n_samples_save),
            facecolor = (1, 1, 1)
        )
    else:
        fig = None
        recon_img = None

    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0, 1)))
    for ii, i in enumerate(range(num_xaxis_subplots - recons_per_sample, num_xaxis_subplots)):
        recon = brain_recons[im][ii]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction", fontweight = 'bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')

    return fig, brain_recons, best_picks, recon_img


def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    # Load data
    test_dl = prepare_data(args)
    num_test = len(test_dl)
    prompts_list = prepare_coco(args)

    # Load autoencoder
    outdir_ae = f'../train_logs/{args.autoencoder_name}'
    ckpt_path = os.path.join(outdir_ae, f'epoch120.pth')

    # Load VD pipeline
    vd_pipe = prepare_VD(args, device)
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    # Load CLIP
    clip_extractor = prepare_CLIP(args, device)

    outdir = f'../train_logs/Clip_test/'
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok = True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # define test range
    test_range = np.arange(num_test)
    if args.test_end is None:
        args.test_end = num_test

    # recon loop
    for val_i, (voxel, img, coco, subj) in enumerate(tqdm(test_dl, total = len(test_range))):
        coco_ids = [coco]
        current_prompts_list = [prompts_list[coco_id] for coco_id in coco_ids]
        captions = [prompts[val_i % 3]['caption'] for prompts in current_prompts_list]

        voxel = torch.mean(voxel, axis = 1).float().to(device)
        img = img.to(device)

        with torch.no_grad():
            text_embedding = clip_extractor.embed_text(captions)
            image_embedding = clip_extractor.embed_image(img)
            grid, brain_recons, best_picks, recon_img = reconstruction(
                img,
                text_embedding,
                image_embedding,
                clip_extractor,
                unet,
                vae,
                noise_scheduler,
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

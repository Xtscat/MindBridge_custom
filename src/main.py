import os
import sys

import torch
from accelerate import Accelerator

import utils
from models import (Clipper, MindBridge_image, MindBridge_image_GIT,
                    MindBridge_image_sketch, MindBridge_text, MindSingle_image,
                    MindSingle_image_GIT, MindSingle_image_sketch,
                    MindSingle_text)
from nsd_access import NSDAccess
from options import args

torch.backends.cuda.matmul.allow_tf32 = True

# """this codeblock is to use gloo backendif use nccl, remove this codeblock"""
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
# torch.distributed.init_process_group(backend='gloo')


def config_multi_gpu():
    # Multi-GPU config
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage = 2, gradient_clipping = 1.0)
    accelerator = Accelerator(split_batches = False, mixed_precision = 'no')
    accelerator.print("PID of this process =", os.getpid())
    device = accelerator.device
    accelerator.print("device:", device)
    num_devices = torch.cuda.device_count()
    if num_devices == 0: num_devices = 1
    accelerator.print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    accelerator.print(
        "distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =",
        world_size
    )

    return accelerator, device, local_rank


def prepare_coco(args):
    # Preload coco captions
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k, info_type = 'captions')
    print(type(prompts_list))
    print(prompts_list[0])

    print("coco captions loaded.")

    return prompts_list


def prepare_CLIP(args, device):
    print("Using hidden layer CLIP space")

    if not args.norm_embs:
        print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")

    clip_extractor = Clipper(
        args.clip_variant, device = device, hidden_state = True, norm_embs = args.norm_embs
    ).to(device)

    return clip_extractor


def prepare_voxel2clip_text(args, out_dim_image, out_dim_text, device):
    # Prepare voxel2clip
    if args.adapting:
        args.subj_list = args.subj_source + [args.subj_target]

    voxel2clip_kwargs = dict(
        in_dim = args.pool_num,
        out_dim_text = out_dim_text,
        h = args.h_size,
        n_blocks = args.n_blocks,
        subj_list = args.subj_list,
        adapting = args.adapting
    )
    if len(args.subj_list) == 1:  # Single subject does not need "brain builder"
        voxel2clip_kwargs.pop("adapting")  # Single subject does not need "adapting"
        voxel2clip = MindSingle_text(**voxel2clip_kwargs).to(device)
    else:
        voxel2clip = MindBridge_text(**voxel2clip_kwargs).to(device)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)

    return voxel2clip


def prepare_voxel2clip_img(args, out_dim_image, out_dim_text, device):
    # Prepare voxel2clip
    if args.adapting:
        args.subj_list = args.subj_source + [args.subj_target]

    voxel2clip_kwargs = dict(
        in_dim = args.pool_num,
        out_dim_image = out_dim_image,
        h = args.h_size,
        n_blocks = args.n_blocks,
        subj_list = args.subj_list,
        adapting = args.adapting
    )
    if len(args.subj_list) == 1:  # Single subject does not need "brain builder"
        voxel2clip_kwargs.pop("adapting")  # Single subject does not need "adapting"
        voxel2clip = MindSingle_image(**voxel2clip_kwargs).to(device)
    else:
        voxel2clip = MindBridge_image(**voxel2clip_kwargs).to(device)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)

    return voxel2clip


def prepare_voxel2clip_img_sketch(args, out_dim_image_feature_map, out_dim_image_fc, device):
    # Prepare voxel2clip
    if args.adapting:
        args.subj_list = args.subj_source + [args.subj_target]

    voxel2clip_kwargs = dict(
        in_dim = args.pool_num,
        out_dim_image_feature_map = out_dim_image_feature_map,
        out_dim_image_fc = out_dim_image_fc,
        h = args.h_size,
        n_blocks = args.n_blocks,
        subj_list = args.subj_list,
        adapting = args.adapting
    )
    if len(args.subj_list) == 1:  # Single subject does not need "brain builder"
        voxel2clip_kwargs.pop("adapting")  # Single subject does not need "adapting"
        voxel2clip = MindSingle_image_sketch(**voxel2clip_kwargs).to(device)
    else:
        voxel2clip = MindBridge_image_sketch(**voxel2clip_kwargs).to(device)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)

    return voxel2clip


def prepare_voxel2clip_img_GIT(args, out_dim_image, out_dim_text, device):
    # Prepare voxel2clip
    if args.adapting:
        args.subj_list = args.subj_source + [args.subj_target]

    voxel2clip_kwargs = dict(
        in_dim = args.pool_num,
        out_dim_image = out_dim_image,
        h = args.h_size,
        n_blocks = args.n_blocks,
        subj_list = args.subj_list,
        adapting = args.adapting
    )
    if len(args.subj_list) == 1:  # Single subject does not need "brain builder"
        voxel2clip_kwargs.pop("adapting")  # Single subject does not need "adapting"
        voxel2clip = MindSingle_image_GIT(**voxel2clip_kwargs).to(device)
    else:
        voxel2clip = MindBridge_image_GIT(**voxel2clip_kwargs).to(device)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)

    return voxel2clip


def prepare_trainer_fmri_text(args, accelerator, voxel2clip, clip_extractor, prompts_list, device):
    from trainer_fmri_text import (Trainer_fmri_text, Trainer_fmri_text_adapt,
                                   Trainer_fmri_text_bridge,
                                   Trainer_fmri_text_single)
    if args.adapting:
        trainer = Trainer_fmri_text_adapt(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)
    elif len(args.subj_list) == 1:
        trainer = Trainer_fmri_text_single(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)
    else:
        trainer = Trainer_fmri_text_bridge(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)

    return trainer


def prepare_trainer_fmri_img(args, accelerator, voxel2clip, clip_extractor, device):
    from trainer_fmri_img import (Trainer_fmri_image, Trainer_fmri_image_adapt,
                                  Trainer_fmri_image_bridge,
                                  Trainer_fmri_image_single)
    if args.adapting:
        trainer = Trainer_fmri_image_adapt(args, accelerator, voxel2clip, clip_extractor, device)
    elif len(args.subj_list) == 1:
        trainer = Trainer_fmri_image_single(args, accelerator, voxel2clip, clip_extractor, device)
    else:
        trainer = Trainer_fmri_image_bridge(args, accelerator, voxel2clip, clip_extractor, device)

    return trainer


def prepare_trainer_fmri_vae(args, accelerator, voxel2vae, vae_extractor, device):
    from trainer_fmri_vae import (Trainer_fmri_vae, Trainer_fmri_vae_bridge,
                                  Trainer_fmri_vae_single)
    if args.adapting:
        trainer = Trainer_fmri_vae_adapt(args, accelerator, voxel2vae, vae_extractor, device)
    elif len(args.subj_list) == 1:
        trainer = Trainer_fmri_vae_single(args, accelerator, voxel2vae, vae_extractor, device)
    else:
        trainer = Trainer_fmri_vae_bridge(args, accelerator, voxel2vae, vae_extractor, device)

    return trainer


def prepare_trainer_fmri_img_sketch(args, accelerator, voxel2img, clip_extractor, device):
    from trainer_fmri_img_sketch import (Trainer_fmri_image,
                                         Trainer_fmri_image_adapt,
                                         Trainer_fmri_image_bridge,
                                         Trainer_fmri_image_single)
    if args.adapting:
        trainer = Trainer_fmri_image_adapt(args, accelerator, voxel2img, clip_extractor, device)
    elif len(args.subj_list) == 1:
        trainer = Trainer_fmri_image_single(args, accelerator, voxel2img, clip_extractor, device)
    else:
        trainer = Trainer_fmri_image_bridge(args, accelerator, voxel2img, clip_extractor, device)

    return trainer


def prepare_trainer_fmri_img_GIT(args, accelerator, voxel2img, clip_extractor, device):
    from trainer_fmri_img_GIT import (Trainer_fmri_image,
                                      Trainer_fmri_image_adapt,
                                      Trainer_fmri_image_bridge,
                                      Trainer_fmri_image_single)
    if args.adapting:
        trainer = Trainer_fmri_image_adapt(args, accelerator, voxel2img, clip_extractor, device)
    elif len(args.subj_list) == 1:
        trainer = Trainer_fmri_image_single(args, accelerator, voxel2img, clip_extractor, device)
    else:
        trainer = Trainer_fmri_image_bridge(args, accelerator, voxel2img, clip_extractor, device)

    return trainer


def main():
    accelerator, device, local_rank = config_multi_gpu()
    if local_rank != 0:  # suppress print for non-local_rank=0
        sys.stdout = open(os.devnull, 'w')

    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(args.seed, cudnn_deterministic = False)

    # learning rate will be changed by "acclerate" based on number of processes(GPUs)
    args.max_lr *= accelerator.num_processes

    # backend_os = os.getenv('PL_TORCH_DISTRIBUTED_BACKEND')
    # backend_pytorch = torch.distributed.get_backend()
    # print(f"os backend: {backend_os}")
    # print(f"pytorch backend: {backend_pytorch}")

    # Init Trainer
    if args.trainer_select == "trainer_fmri_text":
        prompts_list = prepare_coco(args)
        clip_extractor = prepare_CLIP(args, device)
        args.clip_variant = "ViT-L/14"
        voxel2clip = prepare_voxel2clip_text(args, None, 77 * 768, device)
        trainer = prepare_trainer_fmri_text(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)
        # trainer.prepare_wandb(local_rank, args)
        trainer.prepare_multi_gpu()
    elif args.trainer_select == "trainer_fmri_img":
        clip_extractor = prepare_CLIP(args, device)
        args.clip_variant = "ViT-L/14"
        voxel2clip = prepare_voxel2clip_img(args, 257 * 768, None, device)
        trainer = prepare_trainer_fmri_img(args, accelerator, voxel2clip, clip_extractor, device)
        # trainer.prepare_wandb(local_rank, args)
        trainer.prepare_multi_gpu()
    elif args.trainer_select == "trainer_fmri_img_sketch":
        args.clip_variant = "ViT-B/32"
        clip_extractor = prepare_CLIP(args, device)
        voxel2clip = prepare_voxel2clip_img_sketch(
            args = args, out_dim_image_feature_map = 50 * 768, out_dim_image_fc = 50 * 512, device = device
        )
        trainer = prepare_trainer_fmri_img_sketch(args, accelerator, voxel2clip, clip_extractor, device)
        # trainer.prepare_wandb(local_rank, args)
        trainer.prepare_multi_gpu()
    elif args.trainer_select == "trainer_fmri_img_GIT":
        args.clip_variant = "GIT-ViT"
        clip_extractor = prepare_CLIP(args, device)
        # voxel2clip = prepare_voxel2clip_img_GIT(args, 257 * 1024, None, device) # for git-large
        voxel2clip = prepare_voxel2clip_img_GIT(args, 197 * 768, None, device)  # for git-base
        trainer = prepare_trainer_fmri_img_GIT(args, accelerator, voxel2clip, clip_extractor, device)
        # trainer.prepare_wandb(local_rank, args)
        trainer.prepare_multi_gpu()

    # # Resume or Load ckpt
    # if args.resume:
    #     trainer.resume()
    # elif args.load_from:
    #     trainer.load()
    # else:  # the ckpt folder should not contain any ckpt. If it contains something, which means you forget to change experiment name, and will cause overwrite.
    #     file_count = len(os.listdir(trainer.outdir))
    #     if file_count > 0:
    #         raise RuntimeError(
    #             "The folder is not empty, please check to avoid overwriting! \n {}\n".format(trainer.outdir)
    #         )

    # Train or Adapt
    trainer.train(local_rank)

    print("\n===Finished!===\n")


if __name__ == '__main__':
    main()

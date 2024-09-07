import os

import evaluate
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

import data
import utils
from modeling_git import GitForCausalLMClipEmb
from models import MindSingle_image_GIT
from nsd_access import NSDAccess
from options import args


def prepare_GIT(args, device):
    """
        captions_from_brain vs captions_from_image:
            train with softclip loss:
                'bleu@1': 0.4699310115383261
                'bleu@4': 0.16991790726508074
            train with infonce loss:
                'bleu@1': 0.48482438713707393,
                'bleu@4': 0.1763788201694267,
    """
    # processor = AutoProcessor.from_pretrained("microsoft/git-large-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")
    #
    # git_model = GitForCausalLMClipEmb.from_pretrained(
    #     "microsoft/git-large-coco", cache_dir = "/media/SSD_1_2T/xt/weights/"
    # ).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")

    git_model = GitForCausalLMClipEmb.from_pretrained(
        "microsoft/git-base-coco", cache_dir = "/media/SSD_1_2T/xt/weights/"
    ).to(device)
    vision_model = git_model.git.image_encoder.to(device)
    return processor, git_model, vision_model


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


def prepare_voxel2clip_img_GIT(args, out_dim_image_feature_map, out_dim_text, device):
    voxel2clip_kwargs = dict(
        in_dim = args.pool_num,
        out_dim_image_feature_map = out_dim_image_feature_map,
        h = args.h_size,
        n_blocks = args.n_blocks,
        subj_list = args.subj_load
    )

    # only need to load Single-subject version of MindBridge
    voxel2clip = MindSingle_image_GIT(**voxel2clip_kwargs)

    outdir = f'/media/SSD_1_2T/xt/MindBridge/train_logs/{args.model_name}'
    ckpt_path = os.path.join(outdir, f'{args.ckpt_from}.pth')
    print("ckpt_path", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location = 'cpu')
    print("EPOCH: ", checkpoint['epoch'])
    state_dict = checkpoint['model_state_dict']

    voxel2clip.load_state_dict(state_dict, strict = False)
    voxel2clip.requires_grad_(False)
    voxel2clip.eval().to(device)

    return voxel2clip


def evaluate_captions(predictions: list = None, references: list = None, evaluate_type: str = None):
    assert evaluate_type in [
        'bleu', 'meteor', 'sentence', 'clip'
    ], "evaluate_type must be in ['bleu', 'meteor', 'sentence', 'clip']"

    metrics = {}
    print("Computing metrics...")
    if evaluate_type == 'bleu':
        bleu = evaluate.load('bleu')
        metrics['bleu@1'] = bleu.compute(predictions = predictions, references = references, max_order = 1)
        metrics['bleu@4'] = bleu.compute(predictions = predictions, references = references, max_order = 4)
    elif evaluate_type == 'meteor':
        metor = evaluate.load('meteor')
        metrics['meteor'] = metor.compute(predictions = predictions, references = references)
    elif evaluate_type == 'sentence':
        sentence_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2', cache_folder = "/media/SSD_1_2T/xt/weights/"
        )
        with torch.no_grad():
            embedding_predictions = sentence_model.encode(predictions, convert_to_tensor = True)
            embedding_references = sentence_model.encode(references, convert_to_tensor = True)

            metrics = torch.cosine_similarity(embedding_predictions, embedding_references).mean().item()
    elif evaluate_type == 'clip':
        model_clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir = "/media/SSD_1_2T/xt/weights/"
        )
        processor_clip = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir = "/media/SSD_1_2T/xt/weights/"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir = "/media/SSD_1_2T/xt/weights/"
        )
        with torch.no_grad():
            input_ids = tokenizer(predictions, return_tensors = "pt", padding = True)
            embedding_predictions = model_clip.get_text_features(**input_ids)

            input_ids = tokenizer(references, return_tensors = "pt", padding = True)
            embedding_references = model_clip.get_text_features(**input_ids)

            metrics = torch.cosine_similarity(embedding_predictions, embedding_references).mean().item()

    return metrics


def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    # Load data
    test_dl = prepare_data(args)
    prompts_list = prepare_coco(args)
    num_test = len(test_dl)

    # load voxel2clip
    # voxel2clip = prepare_voxel2clip_img_GIT(args, 257 * 1024, None, device) # for git-large
    voxel2clip = prepare_voxel2clip_img_GIT(args, 197 * 768, None, device)  # for git-base

    # load GIT models
    processor, git_model, vision_model = prepare_GIT(args, device)

    # outdir = f'../train_logs/{args.model_name}_svg'
    # save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    # os.makedirs(save_dir, exist_ok = True)
    # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # define test range
    test_range = np.arange(num_test)
    if args.test_end is None:
        args.test_end = num_test

    # recon loop
    captions_from_brain = []
    captions_from_coco = []
    captions_from_image = []
    for val_i, (voxel, img, img_lowlevel, coco, subj) in enumerate(tqdm(test_dl, total = len(test_range))):
        repeat_index = val_i % 3  # randomly choose the one in the repeated three
        if val_i < args.test_start:
            continue
        if val_i >= args.test_end:
            break
        if (args.samples is not None) and (val_i not in args.samples):
            continue

        coco_ids = coco.squeeze().tolist()
        coco_ids = [coco_ids]
        current_prompts_list = [prompts_list[coco_id] for coco_id in coco_ids]
        coco_captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]
        # print(f"original captions: {captions}")
        voxel = torch.mean(voxel, axis = 1).float().to(device)
        img = img.to(device)

        with torch.no_grad():
            # image_embeds = voxel2clip(voxel).reshape(1, -1, 1024) # for git-large
            image_embeds = voxel2clip(voxel).reshape(1, -1, 768)  # for git-base

            brain_generate_ids = git_model.generate(pixel_values = image_embeds, max_length = 50)
            brain_generated_caption = processor.batch_decode(brain_generate_ids, skip_special_tokens = True)

            image_pixel = processor(images = img, return_tensors = "pt").pixel_values.to(device)
            image_embedding = vision_model(image_pixel).last_hidden_state
            image_generate_ids = git_model.generate(pixel_values = image_embedding, max_length = 50)
            image_generate_caption = processor.batch_decode(image_generate_ids, skip_special_tokens = True)

            captions_from_brain += brain_generated_caption
            captions_from_coco += coco_captions
            captions_from_image += image_generate_caption

            # # save captions_from_brain
            # file_path = img_lowlevel[0].rsplit('.', 1)[0].rsplit('.', 1)[0]
            # byte_data = np.frombuffer(brain_generated_caption[0].encode(), dtype = np.uint8)
            # np.save(f"{file_path}.caption.npy", byte_data)
            #
            # if val_i % 10 == 0:
            #     print(
            #         f"captions_from_image: {image_generate_caption} ==> captions_from_brain: {brain_generated_caption}"
            #     )

        data = {
            'captions_from_coco': captions_from_coco,
            'captions_from_image': captions_from_image,
            'captions_from_brain': captions_from_brain
        }
        df = pd.DataFrame(data)
        df.to_excel('captions.xlsx')

    # compute text metrics
    metrics_bleu = evaluate_captions(
        predictions = captions_from_brain, references = captions_from_image, evaluate_type = 'bleu'
    )
    metrics_meteor = evaluate_captions(
        predictions = captions_from_brain, references = captions_from_image, evaluate_type = 'meteor'
    )
    metrics_sentence = evaluate_captions(
        predictions = captions_from_brain, references = captions_from_image, evaluate_type = 'sentence'
    )
    metrics_clip = evaluate_captions(
        predictions = captions_from_brain, references = captions_from_image, evaluate_type = 'clip'
    )

    data = {
        "Metric": ["Bleu@1", "Bleu@4", "Meteor", "Sentence", "CLIP"],
        "Value": [
            metrics_bleu['bleu@1']['bleu'], metrics_bleu['bleu@4']['bleu'], metrics_meteor['meteor']['meteor'],
            metrics_sentence, metrics_clip
        ],
    }
    title = "brain captions and image captions"
    print(title)
    df = pd.DataFrame(data)
    print(df.to_string(index = False))


if __name__ == "__main__":
    utils.seed_everything(seed = args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # args.model_name = "MindBrige_image_GIT_ViT_infonce_token"
    args.model_name = "MindBrige_image_GIT_ViT_infonce_largemse"
    # args.model_name = "MindBrige_image_GIT_ViT-softclip"
    # args.model_name = "MindBrige_image_GIT_ViT-L_14_infonce"
    args.ckpt_from = "last"
    args.h_size = 2048
    args.n_blocks = 4
    args.pool_type = "max"
    args.subj_load = [1]
    args.subj_test = 1
    args.pool_num = 8192

    main(device)

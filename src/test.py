import pandas as pd
import torch

import data
import utils
from nsd_access import NSDAccess
from options import args


def prepare_coco(args):
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k, info_type = 'captions')

    print("coco captions loaded.")

    return prompts_list


def prepare_data(self):
    train_dl, val_dl = data.get_dls(
        subject = args.subj_list[0],
        data_path = args.data_path,
        batch_size = args.batch_size,
        val_batch_size = args.val_batch_size,
        num_workers = args.num_workers,
        pool_type = args.pool_type,
        pool_num = args.pool_num,
        length = args.length,
        seed = args.seed,
    )
    return train_dl


def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    # Load data
    train_dl = prepare_data(args)
    prompts_list = prepare_coco(args)

    # recon loop
    captions_from_coco = []
    for train_i, data_i in enumerate(train_dl):
        repeat_index = train_i % 3
        _, coco, _ = data_i
        coco_ids = coco.squeeze().tolist()
        if type(coco_ids) == int:
            coco_ids = [coco_ids]
        current_prompt_list = [prompts_list[coco_id] for coco_id in coco_ids]
        captions = [prompts[repeat_index]['caption'] for prompts in current_prompt_list]
        captions_from_coco += captions

        data = {'captions_from_coco': captions_from_coco, }
        df = pd.DataFrame(data)
        df.to_excel(f'subj_{args.subj_list[0]}_captions.xlsx')


if __name__ == "__main__":
    utils.seed_everything(seed = args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.subj_list = [1]
    args.pool_num = 8192
    args.batch_size = 1

    main(device)

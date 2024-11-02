import json
import os
import os.path as op

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm

import data


class NSDAccess(object):

    """
    Class to access NSD dataset and COCO annotations.
    """
    def __init__(self, nsd_folder):
        self.nsd_folder = nsd_folder
        self.nsddata_folder = op.join(self.nsd_folder, 'nsddata')
        self.ppdata_folder = op.join(self.nsd_folder, 'nsddata', 'ppdata')
        self.nsddata_betas_folder = op.join(self.nsd_folder, 'nsddata_betas', 'ppdata')

        self.stimuli_description_file = op.join(
            self.nsd_folder, 'nsddata', 'experiments', 'nsd', 'nsd_stim_info_merged.csv'
        )
        self.coco_annotation_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', '{}_{}.json'
        )

        # Load captions for train and val
        self.coco_captions_file_train = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', 'captions_train2017.json'
        )
        self.coco_captions_file_val = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', 'captions_val2017.json'
        )

    def read_image_annotations(self, image_indices, subjects):
        """Get COCO categories, bounding boxes, and captions for a list of image indices for specified subjects."""
        if not hasattr(self, 'stim_descriptions'):
            self.stim_descriptions = pd.read_csv(self.stimuli_description_file, index_col = 0)

        image_annotations = {subject: {'train': [], 'val': []} for subject in subjects}
        subject_counts = {subject: {'train': 0, 'val': 0} for subject in subjects}

        # Load COCO annotations
        annot_file_train = self.coco_annotation_file.format('instances', 'train2017')
        annot_file_val = self.coco_annotation_file.format('instances', 'val2017')

        # Initialize COCO objects
        coco_train = COCO(annot_file_train)
        coco_val = COCO(annot_file_val)

        # Load category names
        coco_categories = coco_train.loadCats(coco_train.getCatIds())
        category_names = {cat['id']: cat['name'] for cat in coco_categories}

        with open(self.coco_captions_file_train, 'r') as f:
            coco_captions_data_train = json.load(f)
            self.coco_captions_train = {
                str(ann['image_id']): ann['caption'] for ann in coco_captions_data_train['annotations']
            }

        with open(self.coco_captions_file_val, 'r') as f:
            coco_captions_data_val = json.load(f)
            self.coco_captions_val = {
                str(ann['image_id']): ann['caption'] for ann in coco_captions_data_val['annotations']
            }

        # Iterate through image indices
        for image in tqdm(image_indices, desc = "Processing Images"):
            subj_info = self.stim_descriptions.iloc[image]
            coco_id = subj_info['cocoId']
            image_info = {
                'image_id': coco_id,
                'bboxes': [],
                'categories': [],
                'captions': []  # Store captions here
            }

            # Check which subjects are using this cocoId
            for subject in subjects:
                if subj_info[subject] == 1:  # Subject is using this cocoId
                    if subj_info['cocoSplit'] == 'train2017':
                        ann_ids = coco_train.getAnnIds(imgIds = coco_id)
                        anns = coco_train.loadAnns(ann_ids)
                        for ann in anns:
                            image_info['bboxes'].append(ann['bbox'])
                            image_info['categories'].append(category_names[ann['category_id']])
                        # Extract captions
                        image_info['captions'] = self.coco_captions_train.get(
                            str(coco_id), []
                        )  # Get captions for this coco_id
                        image_annotations[subject]['train'].append(image_info)
                        subject_counts[subject]['train'] += 1

                    elif subj_info['cocoSplit'] == 'val2017':
                        ann_ids = coco_val.getAnnIds(imgIds = coco_id)
                        anns = coco_val.loadAnns(ann_ids)
                        for ann in anns:
                            image_info['bboxes'].append(ann['bbox'])
                            image_info['categories'].append(category_names[ann['category_id']])
                        # Extract captions
                        image_info['captions'] = self.coco_captions_val.get(
                            str(coco_id), []
                        )  # Get captions for this coco_id
                        image_annotations[subject]['val'].append(image_info)
                        subject_counts[subject]['val'] += 1

        return image_annotations, subject_counts


def prepare_coco_annotations(args):
    # Initialize NSDAccess and get the annotations for specified subjects
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, 73000))  # Image indices from 0 to 72,999
    subjects = ['subject1', 'subject2', 'subject5', 'subject7']
    annotations_dict, subject_counts = nsda.read_image_annotations(coco_73k, subjects)

    print("COCO annotations loaded.")
    return annotations_dict, subject_counts


def convert_numpy_to_python(obj):
    """
    Convert numpy data types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


def save_annotations_to_json(annotations_dict, subject_counts, output_dir):
    """Save the annotations for each subject to a JSON file and print counts."""
    for subject, annotations in annotations_dict.items():
        # Save train annotations
        output_file_train = op.join(output_dir, f"{subject}_train_annotations.json")
        annotations_train_converted = convert_numpy_to_python(annotations['train'])
        with open(output_file_train, 'w') as f:
            json.dump(annotations_train_converted, f, indent = 4)
        print(f"COCO train annotations for {subject} saved to {output_file_train}.")
        print(f"Number of train images for {subject}: {subject_counts[subject]['train']}")

        # Save validation annotations
        output_file_val = op.join(output_dir, f"{subject}_val_annotations.json")
        annotations_val_converted = convert_numpy_to_python(annotations['val'])
        with open(output_file_val, 'w') as f:
            json.dump(annotations_val_converted, f, indent = 4)
        print(f"COCO val annotations for {subject} saved to {output_file_val}.")
        print(f"Number of val images for {subject}: {subject_counts[subject]['val']}")

        # Save captions separately
        captions_train = [item['captions'] for item in annotations['train']]
        captions_val = [item['captions'] for item in annotations['val']]

        captions_train_file = op.join(output_dir, f"{subject}_train_captions.json")
        with open(captions_train_file, 'w') as f:
            json.dump(captions_train, f, indent = 4)
        print(f"COCO train captions for {subject} saved to {captions_train_file}.")
        print(f"Number of train images for {subject}: {subject_counts[subject]['train']}")

        captions_val_file = op.join(output_dir, f"{subject}_val_captions.json")
        with open(captions_val_file, 'w') as f:
            json.dump(captions_val, f, indent = 4)
        print(f"COCO val captions for {subject} saved to {captions_val_file}.")
        print(f"Number of train images for {subject}: {subject_counts[subject]['val']}")


def prepare_data(args):
    train_dl, val_dl = data.get_dls(
        subject = args.subj_list[0],
        data_path = "/media/HDD_1_2T/xt/data/natural-scenes-dataset/",
        batch_size = args.batch_size,
        val_batch_size = args.val_batch_size,
        num_workers = args.num_workers,
        pool_type = args.pool_type,
        pool_num = args.pool_num,
        length = args.length,
        seed = args.seed,
    )
    return train_dl, val_dl


def main(device):
    args.batch_size = 1

    # Load data
    train_dl, val_dl = prepare_data(args)
    prompts_list, _ = prepare_coco_annotations(args)  # 假设这里也加载了 COCO 注释

    # 处理训练数据
    train_data_from_coco = []
    for train_i, data_i in enumerate(train_dl):
        repeat_index = train_i % 3
        _, coco, _ = data_i
        coco_ids = coco.squeeze().tolist()
        if type(coco_ids) == int:
            coco_ids = [coco_ids]

        for coco_id in coco_ids:
            current_prompt = prompts_list[coco_id]
            caption = current_prompt[repeat_index]['caption']
            category = current_prompt[repeat_index]['category']
            bbox = current_prompt[repeat_index]['bboxes']

            train_data_from_coco.append({'caption': caption, 'category': category, 'bbox': bbox})

    # 保存训练数据为 JSON
    with open('/media/SSD_1_2T/xt/MindBridge/sub07_train_coco_data.json', 'w') as f:
        json.dump(train_data_from_coco, f, indent = 4)
    print("训练数据已保存为 JSON 文件。")

    # 处理验证数据
    val_data_from_coco = []
    for val_i, data_i in enumerate(val_dl):
        repeat_index = val_i % 3
        _, coco, _ = data_i
        coco_ids = coco.squeeze().tolist()
        if type(coco_ids) == int:
            coco_ids = [coco_ids]

        for coco_id in coco_ids:
            current_prompt = prompts_list[coco_id]
            caption = current_prompt[repeat_index]['caption']
            category = current_prompt[repeat_index]['category']
            bbox = current_prompt[repeat_index]['bboxes']

            val_data_from_coco.append({'caption': caption, 'category': category, 'bbox': bbox})

    # 保存验证数据为 JSON
    with open('/media/SSD_1_2T/xt/MindBridge/sub07_val_coco_data.json', 'w') as f:
        json.dump(val_data_from_coco, f, indent = 4)
    print("验证数据已保存为 JSON 文件。")


if __name__ == "__main__":

    class Args:
        data_path = "/media/SSD_1_2T/mzh/NSD"  # 修改为实际的 NSD 数据集路径
        output_dir = "/media/SSD_1_2T/xt/MindBridge/"  # 保存 JSON 文件的目录
        batch_size = 1  # 设置批量大小
        pool_num = 8192  # 池数量
        seed = 42  # 随机种子
        subj_list = [1, 2, 5, 7]  # 受试者列表
        # subj_list = ['subject1', 'subject2', 'subject5', 'subject7']  # 受试者列表
        val_batch_size = 1  # 设置验证批量大小
        num_workers = 4  # 工作线程数量
        length = None
        pool_type = "default"  # 池类型
        extension = ["subj", ]

    args = Args()

    # 准备并保存 COCO 注释
    coco_annotations, subject_counts = prepare_coco_annotations(args)

    # 保存注释到 JSON 文件
    save_annotations_to_json(coco_annotations, subject_counts, args.output_dir)

    # 处理数据并保存训练和验证数据
    main(None)  # 如果需要设备参数，可以替换 None 为实际设备

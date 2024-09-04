import os
import shutil

import cairosvg
from svg2png import svg2png


def get_svg_paths(folder, file_name):
    for root, dirs, files in os.walk(folder):
        if file_name in files:
            return os.path.join(folder, file_name)
        else:
            raise ValueError()


def get_sub_folders(path):
    sub_folders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sub_folders


def svg2img(source_svg, destination_directory):
    if not os.path.exists(destination_directory):
        raise ValueError(f"{destination_directory} not exists")

    for root, dirs, files in os.walk(source_svg):
        dirs.clear()
        for file in files:
            if file == 'best_iter.svg':
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_svg)
                if relative_path == ".":
                    relative_path = os.path.basename(root)
                destination_path = os.path.join(
                    destination_directory, f"{relative_path.replace(os.sep, '_')}.low_level.png"
                )
                # shutil.copyfile(source_path, destination_path)
                cairosvg.svg2png(
                    url = source_path,
                    write_to = destination_path,
                    output_width = 512,
                    output_height = 512,
                    background_color = 'white'
                )
                print(f"Copied {source_path} to {destination_path}")


svg_root = "/media/SSD_1_2T/xt/CLIPasso-main/output_sketches/MindBrige_image_vitb32_multilayer_23456_mse_mae/subj01/"
png_root = "/media/SSD_1_2T/xt/data/natural-scenes-dataset/webdataset_avg_split/test/subj01/"

svg_folder_list = get_sub_folders(svg_root)
svg_folder_list = sorted(svg_folder_list)

for svg_folder in svg_folder_list:
    svg2img(svg_folder, png_root)

print("Subj01 Done!")

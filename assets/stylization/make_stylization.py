import argparse
import os
from os import path as osp
from tqdm import tqdm
from PIL import Image
import torch
from dataloader.megadepth import MegaDepth
from assets.stylization.stylizer import Stylizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config for stylization')
    parser.add_argument('--datadir', type=str, help='the dataset directory')
    parser.add_argument('--phase', type=str, default="train", help='split (train/test)')
    parser.add_argument('--styledir',
                        type=str,
                        default="/data/datasets/style_transfer_amos/styles_sub_10_reduced",
                        help='')
    parser.add_argument('--fast_stylization', type=int, default=1, help='use the faster version')

    args = parser.parse_args()

    dataset = MegaDepth(args, args.phase)
    stylizer = Stylizer(args)
    _, fnames2 = dataset.get_image_pairs()

    st_path = "stylization_fast" if args.fast_stylization else "stylization_accurate"
    out_path_home = osp.join(args.datadir, st_path, args.phase)
    if not osp.isdir(out_path_home):
        os.makedirs(out_path_home)

    n_failures = 0
    with torch.no_grad():
        for content_fname in tqdm(fnames2, desc='# Stylizing content images'):
            # let us define a destination folder
            img_fname = content_fname[content_fname.rindex('/') + 1:]
            remain_path = content_fname[len(osp.join(args.datadir, args.phase)) + 1:content_fname.rindex('/')]
            out_path = osp.join(out_path_home, remain_path)
            out_fname = osp.join(out_path, img_fname)
            if not osp.isdir(out_path):
                os.makedirs(out_path)

            if not osp.isfile(out_fname):
                try:
                    _, stylized_img = stylizer.forward(content_fname)
                    stylized_img.save(out_fname)
                except:
                    print('some error. Exit')
                    im = Image.open(content_fname).convert('RGB')
                    im.save(out_fname)
                    n_failures += 1
    print("Number of failure images: ", n_failures, ". Done")

from __future__ import print_function
import random
import argparse
import torch
import process_stylization
import shutil
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing
from photo_smooth import Propagator
import os
from os import path as osp


SEED = 1984
N_IMGS = 100
FAST = True
SPLIT = "val"
MODEL = "./PhotoWCTModels/photo_wct.pth"
STYLE_PATH = "/data/datasets/style_transfer_amos/styles_sub_10"
CONTENT_PATH = "/data/datasets/phototourism"
OUTPUT_ST_PATH = osp.join(CONTENT_PATH, "style_transfer_all")
STYLES = ["cloudy", "dusk", "mist", "night", "rainy", "snow"]
#STYLES = ["snow"]

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(MODEL))
p_pro = GIFSmoothing(r=35, eps=0.001) if FAST else Propagator()
p_wct.cuda(0)

with open(osp.join(CONTENT_PATH, SPLIT + "_phototourism_ms.txt"), "r") as f:
    content_fnames = [line.rstrip('\n') for line in f]

for style in STYLES:
    print("Style: {:s}".format(style))
    style_fnames = [img for img in os.listdir(osp.join(STYLE_PATH, style)) if img[-3:] in ["png", "jpg"]]
    for style_fname in style_fnames:
        k_cont = 0
        for content_fname in content_fnames:
            scene = content_fname.split('/')[0]
            output_path = osp.join(CONTENT_PATH, "style_transfer_all", scene, style)
            if not osp.isdir(output_path):
                os.makedirs(output_path)

            out_fname = content_fname.split('/')[-1][:-4] + "___" + style_fname[:-4] + ".png"
            if osp.isfile(osp.join(output_path, out_fname)):
                k_cont += 1
                continue

            process_stylization.stylization(p_wct, 
                                            p_pro, 
                                            osp.join(CONTENT_PATH, "train", content_fname),
                                            osp.join(STYLE_PATH, style, style_fname),
                                            [],
                                            [],
                                            osp.join(output_path, out_fname),
                                            1,
                                            False,
                                            False)
            k_cont += 1
            print("style: {:s}, sfname: {:s}, {:d}/{:d}".format(style,
                                                                style_fname,
                                                                k_cont,
                                                                len(content_fnames)))
print("Done!")


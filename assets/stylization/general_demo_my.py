from __future__ import print_function
import random
import time
import argparse
import torch
import assets.stylization.process_stylization as process_stylization
import shutil
from assets.stylization.photo_wct import PhotoWCT
from assets.stylization.photo_gif import GIFSmoothing
from assets.stylization.photo_smooth import Propagator
import os
from os import path as osp


SEED = 1984
FAST = True
MODEL = "./PhotoWCTModels/photo_wct.pth"
STYLE_PATH = osp.join(os.getcwd(), "style-transfer", "night")
OUTPUT_PATH = osp.join(os.getcwd(), "results", "d_tmp")
IMAGE_PATH = osp.join(os.getcwd(), "images", "content_tmp")

if not osp.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(MODEL))
p_pro = GIFSmoothing(r=35, eps=0.001) if FAST else Propagator()
p_wct.cuda(0)

style_imgs = [img for img in os.listdir(STYLE_PATH) if img[-3:] in ["png", "jpg"]]

time_start = time.time()
content_imgs = [fname for fname in os.listdir(IMAGE_PATH) if fname[-3:] == "png"]

i = 0
for content_img in content_imgs:
    for style_img in style_imgs:
        fname_out = osp.join(OUTPUT_PATH, content_img + ".st_" + style_img[:-4] + ".jpg")
        if osp.isfile(fname_out):
            continue
        process_stylization.stylization(p_wct,
                                        p_pro,
                                        osp.join(IMAGE_PATH, content_img),
                                        osp.join(STYLE_PATH, style_img),
                                        [],
                                        [],
                                        fname_out,
                                        1,
                                        False,
                                        False)
    print("Image ", i + 1, " is processed out of ", len(content_imgs))
    i += 1
print("Elapsed time: ", time.time() - time_start)
print("Done")

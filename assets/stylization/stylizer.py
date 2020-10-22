import os
from os import path as osp
import random
import torch
from assets.stylization.process_stylization import stylization_m
from assets.stylization.photo_wct import PhotoWCT
from assets.stylization.photo_gif import GIFSmoothing


class Stylizer(object):
    def __init__(self, styles_path):
        self.styles_path = styles_path
        self.styles = [osp.join(styles_path, img) for img in os.listdir(styles_path)]

        self.p_wct = PhotoWCT()
        self.p_wct.load_state_dict(torch.load("photo_wct.pth"))
        self.p_pro = GIFSmoothing(r=35, eps=0.001)
        self.p_wct.cuda(1)

    def forward(self, content_fname):
        # randomly pick the style
        style_fname = random.choice(self.styles, 1)[0]
        stylized_img = stylization_m(self.p_wct, self.p_pro, content_fname, style_fname)
        return stylized_img

import os
from os import path as osp
import random
import torch
import torch.nn as nn
from assets.stylization.process_stylization import stylization_m
from assets.stylization.photo_wct import PhotoWCT
from assets.stylization.photo_gif import GIFSmoothing


class Stylizer(object):
    def __init__(self, sargs):
        # self.styles = [osp.join(sargs.styledir, img) for img in os.listdir(sargs.styledir)]
        self.styles = []
        for dirpath, dirnames, fnames in os.walk(sargs.styledir):
            self.styles += [osp.join(dirpath, fname) for fname in fnames]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.p_wct = PhotoWCT()
        self.dir_path = osp.dirname(osp.realpath(__file__))
        self.p_wct.load_state_dict(torch.load(osp.join(self.dir_path, "photo_wct.pth")))
        self.fast = sargs.fast_stylization

        '''
        if sargs.multi_gpu:
            print("Use multiple GPUs")
            self.p_wct = nn.DataParallel(self.p_wct)
        '''

        self.p_pro = GIFSmoothing(r=35, eps=0.001)
        self.p_wct.to(device)

    def forward(self, content_fname):
        # randomly pick the style
        style_fname = random.choice(self.styles)
        stylized_arr, stylized_img = stylization_m(self.p_wct, self.p_pro, content_fname, style_fname, self.fast)
        return stylized_arr, stylized_img

import torch
from torch.utils.data import Dataset
import os
from os import path as osp
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
import config
from tqdm import tqdm
from CAPS.caps_model import CAPSModel


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


class HPatchDataset(Dataset):
    def __init__(self, imdir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ])
        self.imfs = []
        for f in os.listdir(imdir):
            scene_dir = os.path.join(imdir, f)
            self.imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])

    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        im_tensor = self.transform(im)
        # using sift keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kpts = sift.detect(gray)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        coord = torch.from_numpy(kpts).float()
        out = {'im': im_tensor, 'coord': coord, 'imf': imf}
        return out

    def __len__(self):
        return len(self.imfs)


def load_caps_features(feat_path, seq_name, img_id):
    data = np.load(osp.join(feat_path, seq_name + "-" + str(img_id) + ".ppm.caps"))
    return data["keypoints"], data["descriptors"]


def benchmark_features(sargs, read_feats):
    lim = [1, 15]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(sargs.extract_img_dir))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(osp.join(sargs.extract_out_dir, sargs.exp_name), seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(osp.join(sargs.extract_out_dir, sargs.exp_name), seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(sargs.extract_img_dir, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def summary(stats):
    n_i = 52
    n_v = 56
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )


if __name__ == '__main__':
    MMA_THRESHOLDS = [1, 3, 5, 7, 10]
    # example code for extracting features for HPatches dataset, SIFT keypoint is used
    args = config.get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = HPatchDataset(args.extract_img_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    model = CAPSModel(args)

    outdir = osp.join(args.extract_out_dir, args.exp_name)
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(data_loader):
            im = data['im'].to(device)
            img_path = data['imf'][0]
            coord = data['coord'].to(device)
            feat_c, feat_f = model.extract_features(im, coord)
            desc = torch.cat((feat_c, feat_f), -1).squeeze(0).detach().cpu().numpy()
            kpt = coord.cpu().numpy().squeeze(0)

            out_path = os.path.join(outdir, '{}-{}'.format(os.path.basename(os.path.dirname(img_path)),
                                                           os.path.basename(img_path),
                                                           ))
            with open(out_path + '.caps', 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=kpt,
                    scores=[],
                    descriptors=desc
                )

        i_err, v_err, stats = benchmark_features(args, load_caps_features)
        summary(stats)

        n_i = 52
        n_v = 56
        for thr in MMA_THRESHOLDS:
            print("MMA@" + str(thr) + " [i]: ", i_err[thr] / (n_i * 5))
            print("MMA@" + str(thr) + " [v]: ", v_err[thr] / (n_v * 5))
            print(11 * '*')


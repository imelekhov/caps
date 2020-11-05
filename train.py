import os
import config
from tensorboardX import SummaryWriter
from CAPS.caps_model import CAPSModel
from dataloader.megadepth import MegaDepthLoader
from utils import cycle
from tqdm import tqdm
import torch
import torch.nn as nn

torch.backends.cudnn.benchmarks = True


def train_megadepth(args):
    # save a copy for the current args in out_folder
    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # tensorboard writer
    tb_log_dir = os.path.join(args.logdir, args.exp_name)
    print('tensorboard log files are stored in {}'.format(tb_log_dir))
    writer = SummaryWriter(tb_log_dir)

    # megadepth data loader
    train_loader = MegaDepthLoader(args).load_data()
    print(len(train_loader))

    test_loader = MegaDepthLoader(args, "test").load_data()

    train_loader_iterator = iter(cycle(train_loader))

    # define model
    model = CAPSModel(args)
    start_step = model.start_step

    # training loop
    val_total_loss = 1e6
    for step in range(start_step + 1, start_step + args.n_iters + 1):
        data = next(train_loader_iterator)
        model.set_input(data)
        model.optimize_parameters()
        model.write_summary(writer, step)

        if step % args.save_interval == 0 and step > 0:
            val_loss = 0.
            for test_sample in tqdm(test_loader):
                model.set_input(test_sample)
                val_loss += model.validate()
            val_loss /= len(test_loader)

            if val_loss < val_total_loss:
                model.save_model(step)
                val_total_loss = val_loss
                print("%s | Step: %d, Loss: %2.5f" % ("val_caps", step, val_total_loss))

            writer.add_scalar('val:total_loss', val_loss, step)


if __name__ == '__main__':
    args = config.get_args()
    train_megadepth(args)

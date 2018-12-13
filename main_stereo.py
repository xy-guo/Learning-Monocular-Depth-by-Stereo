from __future__ import division, print_function
import mkl
mkl.set_num_threads(1)  # to avoid multi-threads in dataloader

import os
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datasets import StereoDataset
from models import StereoModel
from utils.parallel import DataParallelOnlyGatherFirst
from utils.util_functions import *

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='train or test', default='train', choices=["train", "test"])
parser.add_argument('--dataset', type=str, help='dataset', default="sceneflow", choices=["sceneflow", "kitti", "cityscapes"])
parser.add_argument('--data_path', type=str, help='data path', required=True)
parser.add_argument('--train_list', type=str, help='path to the training list')
parser.add_argument('--val_list', type=str, help='path to the validation list')
parser.add_argument('--test_list', type=str, help='path to the testing list')

parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--height', type=int, help='input height', default=384)
parser.add_argument('--width', type=int, help='input width', default=768)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lrepochs', type=str, help='epoch ids when descending the learning rate', default="25,35,45")

parser.add_argument('--loss_type', type=str, help='loss type', default='stereo_sup_w_mask')

# TODO: check the arguments
# parser.add_argument('--lr_loss_weight', type=float, help='left-right consistency weight', default=1.0)
# parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)

parser.add_argument('--work_dir', type=str, help='the directory to save checkpoints and logs', default='./default_work_dir')
parser.add_argument('--load_latest', help='load latest checkpoint from work dir', action='store_true')
parser.add_argument('--load_ckpt', type=str, default="", help='load specific checkpoint')
parser.add_argument('--num_threads', type=int, help='number of threads for data loading', default=8)

# parse and check arguments
args = parser.parse_args()
assert not (args.load_latest and args.load_ckpt)
if not os.path.isdir(args.work_dir):
    os.mkdir(args.work_dir)


# training
def train():
    # some checks
    assert args.train_list and args.val_list

    # model
    model = DataParallelOnlyGatherFirst(StereoModel(args.loss_type)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.)

    # loading model parameters
    starting_epoch = 0
    if args.load_ckpt:
        print("loading checkpoint (--load_ckpt): {}".format(args.load_ckpt))
        ckpt = torch.load(args.load_ckpt)
        model.load_state_dict(ckpt['model'])
        print("loaded")
    elif args.load_latest:
        latest_checkpoint_fn = find_latest_checkpoint(args.work_dir)
        print("loading latest checkpoint in work dir: {}".format(latest_checkpoint_fn))
        ckpt = torch.load(latest_checkpoint_fn)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epoch = ckpt['epoch'] + 1
        print("loaded")

    train_dataset = StereoDataset(args.data_path, args.train_list, args, args.dataset, 'train')
    val_dataset = StereoDataset(args.data_path, args.val_list, args, args.dataset, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_threads,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads,
                                drop_last=False)

    num_training_samples = len(train_dataset)
    steps_per_epoch = len(train_dataloader)
    num_total_steps = args.num_epochs * steps_per_epoch
    print("total number of samples: {}".format(num_training_samples))
    print("total number of steps: {}".format(num_total_steps))
    print("number of trainable parameters: {}".format(count_model_parameters(model)))
    print("change learning rate at epoch {}".format(args.lrepochs))

    print("starting at epoch {}/{}".format(starting_epoch, args.num_epochs))
    logger = SummaryWriter(args.work_dir)


    for epoch_idx in range(starting_epoch, args.num_epochs):
        global_step = len(train_dataloader) * epoch_idx
        adjust_learning_rate(optimizer, epoch_idx, args.learning_rate, args.lrepochs)
        for batch_idx, sample in enumerate(train_dataloader):
            global_step = len(train_dataloader) * epoch_idx + batch_idx
            optimizer.zero_grad()
            loss, scalar_outputs, image_outputs = model(to_cuda_vars(sample))
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                loss_value = loss.detach().cpu().item()
                print('B {}/{} E {}/{} | loss: {:.5f}'.format(epoch_idx, args.num_epochs, batch_idx, len(train_dataloader), loss_value))

                # tensorboard summary
                save_scalars(logger, "train", scalar_outputs, global_step)
                save_images(logger, "train", image_outputs, global_step)

                if args.dataset == "kitti":
                    disps_np = image_outputs["left_disp_est"][0].data.cpu().numpy()[:, 0]
                    save_kitti_metrics(logger, "train", disps_np, sample["left_fn"][:len(disps_np)], global_step)

        torch.save({
            'epoch': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(args.work_dir, 'checkpoint_{:0>6}.ckpt'.format(epoch_idx)))

        model.eval()
        if args.dataset == "kitti":
            test_fns = []
            disp_ests = []
            for batch_idx, sample in enumerate(val_dataloader):
                left_disp_est, _, _ = model(to_cuda_vars(sample))
                test_fns.extend(sample["left_fn"])
                disp_ests.append(left_disp_est.cpu().data.numpy()[:, 0, :, :].copy())
            disp_ests = np.concatenate(disp_ests, 0)
            print("Test | E {}/{}".format(epoch_idx, args.num_epochs))
            save_kitti_metrics(logger, "fullval", disp_ests, test_fns, global_step)
        model.train()


@make_nograd_func
def test():
    # some checks
    assert args.test_list

    # model
    model = DataParallelOnlyGatherFirst(StereoModel(args.loss_type)).cuda()
    model.eval()

    # loading model parameters
    if args.load_ckpt:
        print("loading checkpoint (--load_ckpt): {}".format(args.load_ckpt))
        ckpt = torch.load(args.load_ckpt)
        model.load_state_dict(ckpt['model'])
        print("loaded")
    elif args.load_latest:
        latest_checkpoint_fn = find_latest_checkpoint(args.work_dir)
        print("loading latest checkpoint in work dir: {}".format(latest_checkpoint_fn))
        ckpt = torch.load(latest_checkpoint_fn)
        model.load_state_dict(ckpt['model'])
        print("loaded")

    test_dataset = StereoDataset(args.data_path, args.test_list, args, args.dataset, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads, drop_last=False)

    num_test_samples = len(test_dataset)
    steps_total = len(test_dataloader)
    print("total number of testing samples: {}".format(num_test_samples))
    print("total number of testing steps: {}".format(steps_total))

    disparities = np.zeros((num_test_samples, args.height, args.width), dtype=np.float32)

    EPE = []
    P1 = []
    P3 = []
    P5 = []

    for istep, sample in enumerate(test_dataloader):
        sample_offset = istep * args.batch_size
        num_samples = sample["left"].shape[0]
        print("testing {}/{}, sample {}".format(istep, steps_total, num_samples))
        disp_est, scalar_outputs, _ = model(to_cuda_vars(sample))

        EPE.append(scalar_outputs["EPE"].data[0])
        P1.append(scalar_outputs["P1"].data[0])
        P3.append(scalar_outputs["P3"].data[0])
        P5.append(scalar_outputs["P5"].data[0])
        disp_est = disp_est.data.cpu().numpy()[:, 0, :, :]
        disp_est /= disp_est.shape[2]
        disparities[sample_offset: sample_offset + num_samples] = disp_est

        if istep % 100 == 0:
            print("step {}/{} | EPE {}, P1 {}, P3 {}, P5 {}".format(istep, steps_total, np.mean(EPE), np.mean(P1),
                                                                      np.mean(P3), np.mean(P5)))

    print("Final | EPE {}, P1 {}, P3 {}, P5 {}".format(np.mean(EPE), np.mean(P1), np.mean(P3), np.mean(P5)))
    if args.dataset == "kitti":  # only save for KITTI
        save_fn = os.path.join(args.work_dir, "disparities.npy")
        np.save(save_fn, disparities)
        print("predictions saved to {}".format(save_fn))


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


if __name__ == '__main__':
    main()

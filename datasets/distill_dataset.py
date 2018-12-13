import os
import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from base_dataset import BaseDataset


def read_image_scale(image_path, scale):
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        print("not finding {}".format(image_path))
    img = img.astype(np.float32) / scale
    return img


class DistillDataset(BaseDataset):
    def __init__(self, data_path, filenames_file, args, dataset, mode):
        super(DistillDataset, self).__init__(data_path, filenames_file, dataset, mode, args.height, args.width)
        assert self.dataset in ["kitti", "cityscapes"]

        # self.load_stereo_disp = False
        # self.load_stereo_occmask = False

        # self.disp_dir = "/home/xyguo/dataset/dispnet_disp_{}".format(self.dataset)
        # self.mask_dir = "/home/xyguo/dataset/dispnet_mask_{}".format(self.dataset)

        if self.dataset == "kitti":
            self.stereo_height, self.stereo_width = 384, 1280
        elif self.dataset == "cityscapes":
            self.stereo_height, self.stereo_width = 384, 1280  # TODO: cityscapes


    def __getitem__(self, idx):
        left_fn, right_fn = self.image_file_list[idx]
        left_image_path = os.path.join(self.data_path, left_fn)
        right_image_path = os.path.join(self.data_path, right_fn)
        original_left_image = self.load_image(left_image_path)
        original_right_image = self.load_image(right_image_path)

        left_disp_gt = self.load_disp(left_fn, img_shape=original_left_image.shape)
        right_disp_gt = self.load_disp(right_fn, img_shape=original_right_image.shape)

        left_image, right_image = self.resize_lr_imgs(original_left_image, original_right_image)
        left_disp_gt, right_disp_gt = self.resize_lr_disps(left_disp_gt, right_disp_gt)

        stereo_left_image = self.resize_img(original_left_image, self.stereo_width, self.stereo_height)
        stereo_right_image = self.resize_img(original_right_image, self.stereo_width, self.stereo_height)

        # if self.load_stereo_disp:
        #     left_stereo_disp = read_image_scale(os.path.join(self.disp_dir, left_fn), 65535. * 2.)
        #     right_stereo_disp = read_image_scale(os.path.join(self.disp_dir, right_fn), 65535. * 2.)
        #     left_stereo_disp = cv2.resize(left_stereo_disp, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        #     right_stereo_disp = cv2.resize(right_stereo_disp, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

        # if self.load_stereo_occmask:
        #     left_stereo_occmask = read_image_scale(os.path.join(self.mask_dir, left_fn), 255.)
        #     right_stereo_occmask = read_image_scale(os.path.join(self.mask_dir, right_fn), 255.)
        #     left_stereo_occmask = cv2.resize(left_stereo_occmask, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        #     right_stereo_occmask = cv2.resize(right_stereo_occmask, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

        if self.mode == 'train':
            do_swap = np.random.rand() > 0.5
            do_augment = np.random.rand() > 0.5  # different with stereo dataset

            if do_swap:
                left_image, right_image, left_disp_gt, right_disp_gt = self.augment_swap(left_image, right_image, left_disp_gt, right_disp_gt)
                stereo_left_image, stereo_right_image = cv2.flip(stereo_right_image, 1), cv2.flip(stereo_left_image, 1)
                # if self.load_stereo_disp:
                #     left_stereo_disp, right_stereo_disp = cv2.flip(right_stereo_disp, 1), cv2.flip(left_stereo_disp, 1)
                # if self.load_stereo_occmask:
                #     left_stereo_occmask, right_stereo_occmask = cv2.flip(right_stereo_occmask, 1), cv2.flip(left_stereo_occmask, 1)

            if do_augment:
                left_image, right_image = self.augment_color(left_image, right_image)

        # TODO: left, right fn when swapped ?
        sample = {"left": left_image, "right": right_image,
                  "stereo_left": stereo_left_image, "stereo_right": stereo_right_image,
                  "left_disp_gt": left_disp_gt, "right_disp_gt": right_disp_gt,
                  "left_fn": left_fn, "right_fn": right_fn}

        # if self.load_stereo_disp:
        #     sample["left_stereo_disp"] = left_stereo_disp
        #     sample["right_stereo_disp"] = right_stereo_disp
        # if self.load_stereo_occmask:
        #     sample["left_stereo_occmask"] = left_stereo_occmask
        #     sample["right_stereo_occmask"] = right_stereo_occmask

        return self.convert_to_tensor(sample)

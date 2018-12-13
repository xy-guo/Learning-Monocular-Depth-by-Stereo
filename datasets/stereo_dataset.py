import os
import numpy as np
import torch
import cv2
from base_dataset import BaseDataset


class StereoDataset(BaseDataset):
    def __init__(self, data_path, filenames_file, args, dataset, mode, ret_meta_info=False):
        super(StereoDataset, self).__init__(data_path, filenames_file, dataset, mode, args.height, args.width)
        assert self.dataset in ["sceneflow", "kitti", "cityscapes"]
        self.ret_meta_info = ret_meta_info

        if self.mode != "train":  # full size when testing on KITTI
            if self.dataset == "kitti":
                self.image_width, self.image_height = 1280, 384

    def __getitem__(self, idx):
        left_fn, right_fn = self.image_file_list[idx]
        left_image_path = os.path.join(self.data_path, left_fn)
        right_image_path = os.path.join(self.data_path, right_fn)
        left_image = self.load_image(left_image_path)
        right_image = self.load_image(right_image_path)

        original_left_image = left_image.copy()
        original_right_image = right_image.copy()
        original_height, original_width = left_image.shape[:2]
        original_left_image = self.resize_img(original_left_image, 1280, 384)
        original_right_image = self.resize_img(original_right_image, 1280, 384)

        left_disp_gt = self.load_disp(left_fn, img_shape=left_image.shape)
        right_disp_gt = self.load_disp(right_fn, img_shape=right_image.shape)

        if self.mode == 'train':
            do_swap = np.random.rand() > 0.5
            if do_swap:
                left_image, right_image, left_disp_gt, right_disp_gt = self.augment_swap(left_image, right_image, left_disp_gt, right_disp_gt)
                original_left_image, original_right_image = cv2.flip(original_right_image, 1), cv2.flip(original_left_image, 1)
            ret = self.augment_crop(left_image, right_image, left_disp_gt, right_disp_gt, ret_meta_info=self.ret_meta_info)
            left_image, right_image, left_disp_gt, right_disp_gt = ret[:4]
            if self.ret_meta_info:
                meta_info = ret[4]
            left_image, right_image = self.augment_color(left_image, right_image)
        else:
            left_image, right_image = self.resize_lr_imgs(left_image, right_image)
            left_disp_gt, right_disp_gt = self.resize_lr_disps(left_disp_gt, right_disp_gt)

        sample = {"left": left_image, "right": right_image,
                  "original_left": original_left_image, "original_right": original_right_image,
                  "left_disp_gt": left_disp_gt, "right_disp_gt": right_disp_gt,
                  "left_fn": left_fn, "right_fn": right_fn}

        if self.mode == "train" and self.ret_meta_info:
            sample.update(meta_info)  # "x_off", "y_off", "crop_width", "crop_height"
            sample.update({"original_height": original_height, "original_width": original_width})

        return self.convert_to_tensor(sample)

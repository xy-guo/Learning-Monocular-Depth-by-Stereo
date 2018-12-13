import os
import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from base_dataset import BaseDataset


class MonoDataset(BaseDataset):
    def __init__(self, data_path, filenames_file, args, dataset, mode):
        super(MonoDataset, self).__init__(data_path, filenames_file, dataset, mode, args.height, args.width)

    def __getitem__(self, idx):
        left_fn, right_fn = self.image_file_list[idx]
        left_image_path = os.path.join(self.data_path, left_fn)
        right_image_path = os.path.join(self.data_path, right_fn)
        left_image = self.load_image(left_image_path)
        right_image = self.load_image(right_image_path)

        left_disp_gt = self.load_disp(left_fn, img_shape=left_image.shape)
        right_disp_gt = self.load_disp(right_fn, img_shape=right_image.shape)

        left_image, right_image = self.resize_lr_imgs(left_image, right_image)
        left_disp_gt, right_disp_gt = self.resize_lr_disps(left_disp_gt, right_disp_gt)

        if self.mode == 'train':
            do_swap = np.random.rand() > 0.5
            do_augment = np.random.rand() > 0.5  # different with stereo dataset

            if do_swap:
                left_image, right_image, left_disp_gt, right_disp_gt = self.augment_swap(left_image, right_image, left_disp_gt,
                                                                                         right_disp_gt)
            if do_augment:
                left_image, right_image = self.augment_color(left_image, right_image)

        sample = {"left": left_image, "right": right_image,
                  "left_disp_gt": left_disp_gt, "right_disp_gt": right_disp_gt,
                  "left_fn": left_fn, "right_fn": right_fn}
        return self.convert_to_tensor(sample)

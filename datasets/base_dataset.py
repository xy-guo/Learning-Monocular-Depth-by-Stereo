import os
import re

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils import generate_depth_map, get_focal_length_baseline


def load_kitti_disp(filename, data_root, im_size):
    filename = filename.split()[0]
    splits = filename.split('/')
    camera_id = np.int32(splits[2][-1:])  # 2 is left, 3 is right
    date = splits[0]
    im_id = splits[4][:10]

    vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

    gt_file = os.path.join(data_root, vel)
    gt_calib = os.path.join(data_root, date).rstrip('/') + '/'

    depth = generate_depth_map(gt_calib, gt_file, im_size, camera_id, False, False)
    gt_depth = depth.astype(np.float32)[:, :, np.newaxis]

    valid_mask = depth > 0

    focal_length, baseline = get_focal_length_baseline(gt_calib, camera_id)
    gt_disp = np.zeros_like(gt_depth)
    gt_disp[valid_mask] = (baseline * focal_length) / gt_depth[valid_mask]

    gt_disp[np.logical_not(valid_mask)] = 0
    gt_disp[np.isinf(gt_disp)] = 0

    return gt_disp[:, :, 0]


def load_pfm(file_path):
    file = open(file_path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data).copy()
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    return data, scale


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("not finding {}".format(image_path))
    img = img.astype(np.float32) / 255.

    # if the dataset is cityscapes, we crop the last fifth to remove the car hood
    # if self.dataset == 'cityscapes':
    #     o_height    = tf.shape(image)[0]
    #     crop_height = (o_height * 4) / 5
    #     image  =  image[:crop_height,:,:]

    return img


class BaseDataset(Dataset):
    def __init__(self, data_path, image_list_file, dataset, mode, height, width):
        self.data_path = data_path.rstrip('/')
        self.image_file_list = self.load_file_list(image_list_file)
        self.dataset = dataset
        self.mode = mode
        self.image_height = height
        self.image_width = width

        assert self.mode in ["train", "test", "val"]
        assert self.dataset in ["sceneflow", "kitti", "cityscapes"]
        if self.dataset == "sceneflow":
            self.has_ground_truth = True
            self.sparse_disp = False
        elif self.dataset == "kitti":
            self.has_ground_truth = True
            self.sparse_disp = True
        elif self.dataset == "cityscapes":
            self.has_ground_truth = False
            self.sparse_disp = False  # not used

        self.load_groud_truth = self.has_ground_truth
        print("mode: {}, dataset: {}, load_gt: {}, sparse_gt: {}".format(self.mode, self.dataset, self.load_groud_truth, self.sparse_disp))

        if self.dataset == "sceneflow":
            postfix = "RGB_finalpass"
            assert self.data_path.endswith(postfix)
            self.disp_gt_path = self.data_path[:-len(postfix)] + "disparity"

    def load_file_list(self, filenames_file):
        image_file_list = []
        with open(filenames_file) as f:
            lines = f.readlines()
            for line in lines:
                left_fn, right_fn = line.strip().split()
                image_file_list.append((left_fn, right_fn))
        return image_file_list

    # TODO: check
    def load_image(self, path):
        img = read_image(path)
        if self.dataset == "cityscapes":
            img = img[:img.shape[0] * 4 // 5, :, :]
        return img

    def load_disp(self, path, img_shape=None):
        if self.load_groud_truth:
            if self.dataset == "sceneflow":
                pfm_path = os.path.join(self.disp_gt_path, path[:-3] + "pfm")
                disp_gt, gt_scale = load_pfm(pfm_path)
                return disp_gt
            elif self.dataset == "kitti":
                disp_gt = load_kitti_disp(path, self.data_path, img_shape[:2])
                return disp_gt
        return None

    def augment_color(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = np.random.uniform(0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = np.random.uniform(0.8, 1.2)
        random_colors = np.random.uniform(0.95, 1.05, [1, 1, 3]) * random_brightness
        left_image_aug *= random_colors
        right_image_aug *= random_colors

        # saturate
        left_image_aug = np.clip(left_image_aug, 0, 1)
        right_image_aug = np.clip(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def augment_swap(self, left_image, right_image, left_gt=None, right_gt=None):
        left_image, right_image = cv2.flip(right_image, 1), cv2.flip(left_image, 1)
        if left_gt is not None:
            assert right_gt is not None
            left_gt, right_gt = cv2.flip(right_gt, 1), cv2.flip(left_gt, 1)
        return left_image, right_image, left_gt, right_gt

    # def augment_flip(self, left_image, right_image, left_gt=None, right_gt=None):
    #     left_image, right_image = cv2.flip(left_image, 1), cv2.flip(right_image, 1)
    #     if left_gt is not None:
    #         assert right_gt is not None
    #         left_gt, right_gt = -cv2.flip(left_gt, 1), -cv2.flip(right_gt, 1)
    #     return left_image, right_image, left_gt, right_gt

    def augment_crop(self, left_image, right_image, left_gt=None, right_gt=None, scale_range=(0.65, 1.0), ret_meta_info=False):
        assert isinstance(scale_range, tuple)
        assert len(scale_range) == 2
        min_scale, max_scale = scale_range

        # randomly crop on original images
        if min_scale < 1.0:
            scale = np.random.uniform(min_scale, max_scale)
        else:
            scale = 1.0
        cur_height = min(left_image.shape[0], right_image.shape[0])
        cur_width = min(left_image.shape[1], right_image.shape[1])
        crop_height = int(cur_height * scale)
        crop_width = int(cur_width * scale)

        x_off = np.random.randint(cur_width - crop_width + 1)
        y_off = np.random.randint(cur_height - crop_height + 1)

        left_image = left_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]
        right_image = right_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]

        # resize
        left_image, right_image = self.resize_lr_imgs(left_image, right_image)

        # crop and resize on depth maps
        if left_gt is not None:
            assert right_gt is not None
            left_gt = left_gt[y_off: y_off + crop_height, x_off: x_off + crop_width]
            right_gt = right_gt[y_off: y_off + crop_height, x_off: x_off + crop_width]
            left_gt, right_gt = self.resize_lr_disps(left_gt, right_gt)

        if not ret_meta_info:
            return left_image, right_image, left_gt, right_gt
        else:
            return left_image, right_image, left_gt, right_gt, \
                   {"x_off": x_off, "y_off": y_off, "crop_width": crop_width, "crop_height": crop_height}

    def resize_img(self, img, width, height):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def resize_disp(self, disp, width, height):
        if not self.sparse_disp:
            return cv2.resize(disp, (width, height), interpolation=cv2.INTER_AREA) * (width / float(disp.shape[1]))
        else:
            # sparse disparity map (kitti ground truth) can not be resized with interpolation
            assert len(disp.shape) == 2
            resized = np.zeros([height, width], dtype=np.float32)
            hs, ws = np.where(disp > 0)

            ratio = width / float(disp.shape[1])
            resized_hs = (hs * (height / float(disp.shape[0]))).astype(np.int)
            resized_ws = (ws * (width / float(disp.shape[1]))).astype(np.int)
            val_inds = (resized_ws >= 0) & (resized_ws < width)
            val_inds = val_inds & (resized_hs >= 0) & (resized_hs < height)

            hs, ws, resized_hs, resized_ws = hs[val_inds], ws[val_inds], resized_hs[val_inds], resized_ws[val_inds]
            resized[resized_hs, resized_ws] = disp[hs, ws] * ratio

            return resized

    def resize_lr_imgs(self, left_image, right_image):
        left_image = self.resize_img(left_image, self.image_width, self.image_height)
        right_image = self.resize_img(right_image, self.image_width, self.image_height)
        return left_image, right_image

    def resize_lr_disps(self, left_disp, right_disp):
        if left_disp is None and right_disp is None:
            return None, None
        left_disp = self.resize_disp(left_disp, self.image_width, self.image_height)
        right_disp = self.resize_disp(right_disp, self.image_width, self.image_height)
        return left_disp, right_disp

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        return NotImplemented

    def convert_to_tensor(self, sample):
        tensor_sample = {}
        for k, v in sample.items():
            if v is None:
                continue
            elif isinstance(v, str):
                tensor_sample[k] = v
            elif isinstance(v, np.ndarray):
                if len(v.shape) == 3:
                    tensor_sample[k] = torch.from_numpy(np.transpose(v, [2, 0, 1]))
                else:
                    tensor_sample[k] = torch.from_numpy(v.copy()[np.newaxis, :, :])
            elif isinstance(v, (float, int)):
                tensor_sample[k] = v
            else:
                raise NotImplemented
        return tensor_sample

import torch
from func.correlation1d_package.correlation1d import Correlation1d
from func.resample1d_package.resample1d import Resample1dFunction
from model_utils import *
from utils.util_functions import unsqueeze_dim0_tensor


def Conv2dBlock2(c_in, c_out, k_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k_size, stride, padding),
        nn.LeakyReLU(0.1),
        nn.Conv2d(c_out, c_out, k_size, 1, padding),
        nn.LeakyReLU(0.1)
    )


def Conv2dBlock1(c_in, c_out, k_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k_size, stride, padding),
        nn.LeakyReLU(0.1)
    )


class StereoNet(nn.Module):
    def __init__(self):
        super(StereoNet, self).__init__()
        self.do_corr = True
        self.encoder_use_bn = False

        self.conv1 = Conv2dBlock1(3, 64, 7, 2, 3)
        self.conv2 = Conv2dBlock1(64, 128, 5, 2, 2)
        if self.do_corr:
            self.corr = Correlation1d(pad_size=40, kernel_size=1, max_displacement=40, stride1=1, stride2=1,
                                      corr_multiply=1)
            self.conv_redir = Conv2dBlock1(128, 64, 1, 1, 0)
        self.conv3 = Conv2dBlock2(64 + 41 if self.do_corr else 128, 256, 3, 2, 1)
        self.conv4 = Conv2dBlock2(256, 512, 3, 2, 1)
        self.conv5 = Conv2dBlock2(512, 512, 3, 2, 1)
        self.conv6 = Conv2dBlock2(512, 1024, 3, 2, 1)

        self.upconv5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, 0)
        self.iconv5 = nn.Conv2d(512 + 512, 512, 3, 1, 1)

        self.upconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1, 0)
        self.iconv4 = nn.Conv2d(256 + 512, 256, 3, 1, 1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, 0)
        self.iconv3 = nn.Conv2d(128 + 256, 128, 3, 1, 1)
        self.disp3 = nn.Conv2d(128, 1, 3, 1, 1)
        self.occmask3 = nn.Conv2d(128, 1, 3, 1, 1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, 0)
        self.iconv2 = nn.Conv2d(64 + 128, 64, 3, 1, 1)
        self.disp2 = nn.Conv2d(64, 1, 3, 1, 1)
        self.occmask2 = nn.Conv2d(64, 1, 3, 1, 1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1, 0)
        self.iconv1 = nn.Conv2d(32 + 64, 32, 3, 1, 1)
        self.disp1 = nn.Conv2d(32, 1, 3, 1, 1)
        self.occmask1 = nn.Conv2d(32, 1, 3, 1, 1)

        self.upconv0 = nn.ConvTranspose2d(32, 16, 4, 2, 1, 0)
        self.iconv0 = nn.Conv2d(16, 16, 3, 1, 1)
        self.disp0 = nn.Conv2d(16, 1, 3, 1, 1)
        self.occmask0 = nn.Conv2d(16, 1, 3, 1, 1)

        initilize_modules(self.modules())

    def forward(self, x1, x2):
        if self.do_corr:
            conv1a = self.conv1(x1)
            conv1b = self.conv1(x2)
            conv2a = self.conv2(conv1a)
            conv2b = self.conv2(conv1b)
            corr = F.leaky_relu(self.corr(conv2a, conv2b), 0.1)
            conv_redir = self.conv_redir(conv2a)
            corr_and_redir = torch.cat([conv_redir, corr], dim=1)

            conv3 = self.conv3(corr_and_redir)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)

            skip1 = conv1a
            skip2 = conv2a
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
        else:
            conv1 = self.conv1(torch.cat([x1, x2], dim=1))
            conv2 = self.conv2(conv1)

            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)

            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5

        nonlinear = lambda x: F.leaky_relu(x, 0.1)

        upconv5 = nonlinear(self.upconv5(conv6))  # H/32
        concat5 = torch.cat((upconv5, skip5), 1)
        iconv5 = nonlinear(self.iconv5(concat5))

        upconv4 = nonlinear(self.upconv4(iconv5))  # H/16
        concat4 = torch.cat((upconv4, skip4), 1)
        iconv4 = nonlinear(self.iconv4(concat4))

        upconv3 = nonlinear(self.upconv3(iconv4))  # H/8
        concat3 = torch.cat((upconv3, skip3), 1)
        iconv3 = nonlinear(self.iconv3(concat3))
        disp3 = self.disp3(iconv3)
        occmask3 = self.occmask3(iconv3)
        # disp3up = upsample_nn_nearest(disp3)

        upconv2 = nonlinear(self.upconv2(iconv3))  # H/4
        concat2 = torch.cat((upconv2, skip2), 1)
        iconv2 = nonlinear(self.iconv2(concat2))
        disp2 = self.disp2(iconv2)
        occmask2 = self.occmask2(iconv2)
        # disp2up = upsample_nn_nearest(disp2)

        upconv1 = nonlinear(self.upconv1(iconv2))  # H/2
        concat1 = torch.cat((upconv1, skip1), 1)
        iconv1 = nonlinear(self.iconv1(concat1))
        disp1 = self.disp1(iconv1)
        occmask1 = self.occmask1(iconv1)
        # disp1up = upsample_nn_nearest(disp1)

        upconv0 = nonlinear(self.upconv0(iconv1))
        concat0 = upconv0  # torch.cat((upconv0), 1)
        iconv0 = nonlinear(self.iconv0(concat0))
        disp0 = self.disp0(iconv0)
        occmask0 = self.occmask0(iconv0)

        # TODO: note the changes here
        scale_disps = [disp0 * 20, disp1 * 20, disp2 * 20, disp3 * 20]
        occmasks = [occmask0, occmask1, occmask2, occmask3]
        return scale_disps, occmasks


class StereoSupervisedWithoutOccmaskLoss(nn.Module):
    def __init__(self):
        super(StereoSupervisedWithoutOccmaskLoss, self).__init__()

    def forward(self, disp_ests, left_gt):
        nscales = len(disp_ests)
        assert left_gt is not None

        left_gts = generate_max_pyramid(left_gt)
        left_valid_mask = [(d > 0).float() for d in left_gts]

        l1_loss = sum([(disp_ests[i] / 20. - left_gts[i] / 20.).abs()[left_valid_mask[i] > 0].mean() / (2 ** i) for i in
                       range(nscales)])
        total_loss = l1_loss

        disp_errors = [(disp_ests[i] - left_gts[i]).abs() * left_valid_mask[i] for i in range(nscales)]

        return total_loss, {"total_loss": total_loss, "l1_loss": l1_loss}, \
               {"left_disp_est": disp_ests, "left_disp_error": disp_errors, "left_disp_gt": left_gts}


class StereoSupervisedWithOccmaskLoss(nn.Module):
    def __init__(self):
        super(StereoSupervisedWithOccmaskLoss, self).__init__()

    def generate_image_left(self, right, disp_left):
        return Resample1dFunction()(right, -disp_left)

    def forward(self, disp_ests, occmask_logits, left_gt, right_gt):
        nscales = len(disp_ests)
        assert left_gt is not None
        assert right_gt is not None

        left_gt_warp = self.generate_image_left(right_gt, left_gt)
        occmask_gt = ((left_gt_warp - left_gt).abs() <= 1.0).float()
        occmask_gt = F.max_pool2d(F.pad(occmask_gt, (1, 1, 1, 1), value=1.0), 3, stride=1)

        left_gts = generate_max_pyramid(left_gt)
        occmask_gts = generate_pyramid(occmask_gt)
        del left_gt, occmask_gt  # to avoid bug

        l1_loss = sum([(disp_ests[i] / 20 - left_gts[i] / 20.).abs().mean() / (2 ** i) for i in range(nscales)])
        occmask_loss = sum(
            [F.binary_cross_entropy_with_logits(occmask_logits[i], occmask_gts[i]) / (2 ** i) for i in range(nscales)])

        total_loss = l1_loss + occmask_loss

        # visualization
        occmask_ests = [F.sigmoid(m) for m in occmask_logits]
        occmask_errors = [(occmask_ests[i] - occmask_gts[i]).abs() for i in range(nscales)]
        disp_errors = [(disp_ests[i] - left_gts[i]).abs() for i in range(nscales)]

        return total_loss, {"total_loss": total_loss, "occmask_loss": occmask_loss, "l1_loss": l1_loss}, \
               {"left_occmask_est": occmask_ests, "left_disp_est": disp_ests,
                "left_occmask_error": occmask_errors, "left_disp_error": disp_errors,
                "left_occmask_gt": occmask_gts, "left_disp_gt": left_gts}


class StereoUnsupervisedFinetuneLoss(nn.Module):
    def __init__(self):
        super(StereoUnsupervisedFinetuneLoss, self).__init__()

    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def generate_left(self, right, disp_left):
        return Resample1dFunction()(right, -disp_left)

    def generate_right(self, left, disp_right):
        return Resample1dFunction()(left, disp_right)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # VALID padding
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def forward(self, left_disp_ests, left_image, right_image, left_pt_disp, left_pt_occmask):
        add_weights = lambda lst: [x / (2 ** i) for i, x in enumerate(lst)]

        # left_disp_ests and right_disp_ests should be absolute disparity values
        assert left_pt_occmask.size(1) == 1
        nscales = len(left_disp_ests)

        left_pyramid = generate_pyramid(left_image)
        right_pyramid = generate_pyramid(right_image)

        # pretrained disp
        left_pt_disp = generate_max_pyramid(left_pt_disp)

        # pretrained occlusion mask
        # left_pt_occmask[:, :, :, :int(0.15 * left_pt_occmask.size(3))] = 0
        left_pt_occmask = generate_pyramid((left_pt_occmask >= 0.8).float())

        # warp left/right image and disparity map
        left_pyramid_warped = [self.generate_left(right_pyramid[i], left_disp_ests[i] / 2 ** i) for i in range(nscales)]

        l1_left = [(left_pyramid_warped[i] - left_pyramid[i]).abs() * left_pt_occmask[i] for i in range(nscales)]
        ssim_left = [self.SSIM(left_pyramid_warped[i], left_pyramid[i]) * left_pt_occmask[i][:, :, 1:-1, 1:-1] for i in
                     range(nscales)]

        l1_loss_left = [d.mean() for d in l1_left]
        ssim_loss_left = [s.mean() for s in ssim_left]
        image_loss_left = [0.85 * ssim_loss_left[i] + 0.15 * l1_loss_left[i] for i in range(nscales)]

        # losses
        image_loss = sum(add_weights(image_loss_left))

        sup_l1_left_loss = [((left_disp_ests[i] - left_pt_disp[i]).abs() * ((1. - left_pt_occmask[i]) + 0.1)).mean() for
                            i in range(nscales)]
        sup_loss = sum(add_weights(sup_l1_left_loss)) * 0.025

        sup_diff_left_loss = [((self.gradient_x(left_disp_ests[i]) - self.gradient_x(left_pt_disp[i])).abs().mean() +
                    (self.gradient_y(left_disp_ests[i]) - self.gradient_y(left_pt_disp[i])).abs().mean()) / 2.0 for i in range(nscales)]
        sup_diff_loss = sum(add_weights(sup_diff_left_loss)) * 0.05  # half lambda_1 lambda_2 produce more stable results

        total_loss = image_loss + sup_loss + sup_diff_loss

        return total_loss, \
               {"total_loss": total_loss, "image_loss": image_loss, "sup_loss": sup_loss,
                "sup_diff_loss": sup_diff_loss,
                "image_loss_left": image_loss_left,
                "l1_loss_left": l1_loss_left,
                "ssim_loss_left": ssim_loss_left}, \
               {"left_pyramid": left_pyramid, "right_pyramid": right_pyramid,
                "left_disp_ests": left_disp_ests,
                "left_pyramid_warped": left_pyramid_warped,
                "l1_left": l1_left,
                "ssim_left": ssim_left,
                "left_pretrained_disp": left_pt_disp,
                "left_pretrained_occmask": left_pt_occmask}

    # def forward(self, left_disp_ests, right_disp_ests, left_image, right_image,
    #             left_pt_disp, right_pt_disp, left_pt_occmask, right_pt_occmask):
    #     add_weights = lambda lst: [x / (2 ** i) for i, x in enumerate(lst)]
    #
    #     # left_disp_ests and right_disp_ests should be absolute disparity values
    #     assert left_pt_occmask.size(1) == 1
    #     nscales = len(left_disp_ests)
    #
    #     left_pyramid = generate_pyramid(left_image)
    #     right_pyramid = generate_pyramid(right_image)
    #
    #     # pretrained disp
    #     left_pt_disp = generate_max_pyramid(left_pt_disp)
    #     right_pt_disp = generate_max_pyramid(right_pt_disp)
    #
    #     # pretrained occlusion mask
    #     left_pt_occmask = generate_pyramid((left_pt_occmask >= 0.8).float())
    #     right_pt_occmask = generate_pyramid((right_pt_occmask >= 0.8).float())
    #
    #     # warp left/right image and disparity map
    #     left_pyramid_warped = [self.generate_left(right_pyramid[i], left_disp_ests[i] / 2 ** i) for i in range(nscales)]
    #     right_pyramid_warped = [self.generate_right(left_pyramid[i], right_disp_ests[i] / 2 ** i) for i in
    #                             range(nscales)]
    #
    #     l1_left = [(left_pyramid_warped[i] - left_pyramid[i]).abs() * left_pt_occmask[i] for i in range(nscales)]
    #     l1_right = [(right_pyramid_warped[i] - right_pyramid[i]).abs() * right_pt_occmask[i] for i in range(nscales)]
    #     ssim_left = [self.SSIM(left_pyramid_warped[i], left_pyramid[i]) * left_pt_occmask[i][:, :, 1:-1, 1:-1] for i in
    #                  range(nscales)]
    #     ssim_right = [self.SSIM(right_pyramid_warped[i], right_pyramid[i]) * right_pt_occmask[i][:, :, 1:-1, 1:-1] for i
    #                   in range(nscales)]
    #
    #     l1_loss_left = [d.mean() for d in l1_left]
    #     l1_loss_right = [d.mean() for d in l1_right]
    #     ssim_loss_left = [s.mean() for s in ssim_left]
    #     ssim_loss_right = [s.mean() for s in ssim_right]
    #     image_loss_left = [0.85 * ssim_loss_left[i] + 0.15 * l1_loss_left[i] for i in range(nscales)]
    #     image_loss_right = [0.85 * ssim_loss_right[i] + 0.15 * l1_loss_right[i] for i in range(nscales)]
    #
    #     # losses
    #     image_loss = sum(add_weights(image_loss_left) + add_weights(image_loss_right))
    #
    #     sup_l1_left_loss = [((left_disp_ests[i] - left_pt_disp[i]).abs() * ((1. - left_pt_occmask[i]) + 0.1)).mean() for
    #                         i in range(nscales)]
    #     sup_l1_right_loss = [((right_disp_ests[i] - right_pt_disp[i]).abs() * ((1. - right_pt_occmask[i]) + 0.1)).mean()
    #                          for i in range(nscales)]
    #     sup_loss = sum(add_weights(sup_l1_left_loss) + add_weights(sup_l1_right_loss)) * 0.05
    #
    #     sup_diff_left_loss = [((self.gradient_x(left_disp_ests[i]) - self.gradient_x(left_pt_disp[i])).abs().mean() +
    #                 (self.gradient_y(left_disp_ests[i]) - self.gradient_y(left_pt_disp[i])).abs().mean()) / 2.0 for i in range(nscales)]
    #     sup_diff_right_loss = [((self.gradient_x(right_disp_ests[i]) - self.gradient_x(right_pt_disp[i])).abs().mean() +
    #                 (self.gradient_y(right_disp_ests[i]) - self.gradient_y(right_pt_disp[i])).abs().mean()) / 2.0 for i in range(nscales)]
    #     sup_diff_loss = sum(add_weights(sup_diff_left_loss) + add_weights(sup_diff_right_loss)) * 0.05
    #
    #     total_loss = image_loss + sup_loss + sup_diff_loss
    #
    #     return total_loss, \
    #            {"total_loss": total_loss, "image_loss": image_loss, "sup_loss": sup_loss,
    #             "sup_diff_loss": sup_diff_loss,
    #             "image_loss_left": image_loss_left, "image_loss_right": image_loss_right,
    #             "l1_loss_left": l1_loss_left, "l1_loss_right": l1_loss_right,
    #             "ssim_loss_left": ssim_loss_left, "ssim_loss_right": ssim_loss_right}, \
    #            {"left_pyramid": left_pyramid, "right_pyramid": right_pyramid,
    #             "left_disp_ests": left_disp_ests, "right_disp_ests": right_disp_ests,
    #             "left_pyramid_warped": left_pyramid_warped, "right_pyramid_warped": right_pyramid_warped,
    #             "l1_left": l1_left, "l1_right": l1_right,
    #             "ssim_left": ssim_left, "ssim_right": ssim_right,
    #             "left_pretrained_disp": left_pt_disp, "right_pretrained_disp": right_pt_disp,
    #             "left_pretrained_occmask": left_pt_occmask, "right_pretrained_occmask": right_pt_occmask}


class StereoModel(nn.Module):
    def __init__(self, loss_type="stereo_sup_w_mask", output_occmask=False):
        super(StereoModel, self).__init__()
        self.model = StereoNet()
        self.__losses__ = {
            "stereo_sup_wo_mask": StereoSupervisedWithoutOccmaskLoss,
            "stereo_sup_w_mask": StereoSupervisedWithOccmaskLoss,
            "stereo_unsup_ft": StereoUnsupervisedFinetuneLoss
        }

        self.loss_type = loss_type
        # print("available losses: {}, choosing {}".format(self.__losses__.keys(), self.loss_type))
        self.model_loss = self.__losses__[loss_type]()
        self.output_occmask = output_occmask

    def forward(self, sample):
        # model
        left, right = sample["left"], sample["right"]
        left_disp_ests, left_occmask_logits = self.model(left, right)

        # model loss
        scalar_outputs, image_outputs = {}, {}
        if self.training:
            if self.loss_type == "stereo_sup_w_mask":
                left_disp_gt, right_disp_gt = sample["left_disp_gt"], sample["right_disp_gt"]
                loss, scalar_outputs, image_outputs = self.model_loss(left_disp_ests, left_occmask_logits, left_disp_gt,
                                                                      right_disp_gt)
            elif self.loss_type == "stereo_sup_wo_mask":
                left_disp_gt = sample["left_disp_gt"]
                loss, scalar_outputs, image_outputs = self.model_loss(left_disp_ests, left_disp_gt)
            elif self.loss_type == "stereo_unsup_ft":
                # right_disp_ests = [fliplr(d) for d in self.model(fliplr(right), fliplr(left))[0]]
                # left_pretrained_disp = sample["left_pretrained_disp_est"]
                # left_pretrained_occmask = sample["left_pretrained_occmask_est"]
                # right_pretrained_disp = sample["right_pretrained_disp_est"]
                # right_pretrained_occmask = sample["right_pretrained_occmask_est"]
                # loss, scalar_outputs, image_outputs = self.model_loss(left_disp_ests, right_disp_ests,
                #                                                       left, right,
                #                                                       left_pretrained_disp, right_pretrained_disp,
                #                                                       left_pretrained_occmask, right_pretrained_occmask)
                left_pretrained_disp = sample["left_pretrained_disp_est"]
                left_pretrained_occmask = sample["left_pretrained_occmask_est"]
                loss, scalar_outputs, image_outputs = self.model_loss(left_disp_ests, left, right,
                                                                      left_pretrained_disp, left_pretrained_occmask)
            else:
                return NotImplemented

        # other outputs
        image_outputs["left"] = generate_pyramid(left)
        image_outputs["right"] = generate_pyramid(right)
        image_outputs["left_disp_est"] = left_disp_ests

        if "left_disp_gt" in sample:  # if ground truth is provided
            left_disp_gt = sample["left_disp_gt"]
            left_disp_gt = generate_max_pyramid(left_disp_gt)
            image_outputs["left_disp_errors"] = [(est - gt).abs() * (gt > 0).float() for est, gt in zip(left_disp_ests, left_disp_gt)]
            err = (left_disp_ests[0] - left_disp_gt[0]).abs()[left_disp_gt[0] > 0]
            scalar_outputs["EPE"] = err.mean()
            scalar_outputs["P1"] = (err > 1.).float().mean()
            scalar_outputs["P3"] = (err > 3.).float().mean()
            scalar_outputs["P5"] = (err > 5.).float().mean()

        # return outputs
        if self.training:
            return unsqueeze_dim0_tensor(loss), unsqueeze_dim0_tensor(scalar_outputs), image_outputs
        else:
            if self.output_occmask:
                return (left_disp_ests[0], F.sigmoid(left_occmask_logits[0])), unsqueeze_dim0_tensor(scalar_outputs), image_outputs
            else:
                return left_disp_ests[0], unsqueeze_dim0_tensor(scalar_outputs), image_outputs

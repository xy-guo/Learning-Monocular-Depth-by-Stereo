import torch.cuda

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


def ConvTranspose2dBlock1(c_in, c_out, k_size, stride, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
        nn.LeakyReLU(0.1)
    )


class MonocularVGG16(nn.Module):
    def __init__(self, use_pretrained_weights=False):
        super(MonocularVGG16, self).__init__()
        self.use_pretrained_weights = use_pretrained_weights
        self.only_train_dec = False

        cfg = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']]

        def make_encoder_layers(cfg, in_c, batch_norm=False):
            layers = []
            in_channels = in_c
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        self.conv1 = make_encoder_layers(cfg[0], 3)
        self.conv2 = make_encoder_layers(cfg[1], 64)
        self.conv3 = make_encoder_layers(cfg[2], 128)
        self.conv4 = make_encoder_layers(cfg[3], 256)
        self.conv5 = make_encoder_layers(cfg[4], 512)

        self.upconv4 = ConvTranspose2dBlock1(512, 256, 4, 2, 1, 0)
        self.iconv4 = Conv2dBlock1(256 + 512, 256, 3, 1, 1)

        self.upconv3 = ConvTranspose2dBlock1(256, 128, 4, 2, 1, 0)
        self.iconv3 = Conv2dBlock1(128 + 256, 128, 3, 1, 1)

        self.upconv2 = ConvTranspose2dBlock1(128, 64, 4, 2, 1, 0)
        self.iconv2 = Conv2dBlock1(64 + 128 + 1, 64, 3, 1, 1)

        self.upconv1 = ConvTranspose2dBlock1(64, 32, 4, 2, 1, 0)
        self.iconv1 = Conv2dBlock1(32 + 64 + 1, 32, 3, 1, 1)

        self.upconv0 = ConvTranspose2dBlock1(32, 16, 4, 2, 1, 0)
        self.iconv0 = Conv2dBlock1(16 + 1, 16, 3, 1, 1)

        disp3 = nn.Conv2d(128, 1, 3, 1, 1)
        disp2 = nn.Conv2d(64, 1, 3, 1, 1)
        disp1 = nn.Conv2d(32, 1, 3, 1, 1)
        disp0 = nn.Conv2d(16, 1, 3, 1, 1)

        self.disp3 = disp3
        self.disp2 = disp2
        self.disp1 = disp1
        self.disp0 = disp0

        initilize_modules(self.modules())
        if use_pretrained_weights:
            print("loading pretrained weights downloaded from pytorch.org")
            self.load_vgg_params(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))
        else:
            print("do not load pretrained weights for the monocular model")

    def load_vgg_params(self, params):
        transfer_cfg = {
            "conv1": {0: 0, 2: 2},
            "conv2": {0: 5, 2: 7},
            "conv3": {0: 10, 2: 12, 4: 14},
            "conv4": {0: 17, 2: 19, 4: 21},
            "conv5": {0: 24, 2: 26, 4: 28}
        }

        def load_with_cfg(module, cfg):
            state_dict = {}
            for to_id, from_id in cfg.items():
                state_dict["{}.weight".format(to_id)] = params["features.{}.weight".format(from_id)]
                state_dict["{}.bias".format(to_id)] = params["features.{}.bias".format(from_id)]
            module.load_state_dict(state_dict)

        for module_name, cfg in transfer_cfg.items():
            load_with_cfg(self._modules[module_name], cfg)

    def forward(self, x):
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406])).cuda()
        var = Variable(torch.FloatTensor([0.229, 0.224, 0.225])).cuda()
        x = (x - mean.view(1, -1, 1, 1)) / (var.view(1, -1, 1, 1))

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        if self.use_pretrained_weights and self.only_train_dec:
            conv1 = conv1.detach()
            conv2 = conv2.detach()
            conv3 = conv3.detach()
            conv4 = conv4.detach()
            conv5 = conv5.detach()

        skip1 = conv1
        skip2 = conv2
        skip3 = conv3
        skip4 = conv4

        upconv4 = self.upconv4(conv5)  # H/16
        concat4 = torch.cat((upconv4, skip4), 1)
        iconv4 = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)  # H/8
        concat3 = torch.cat((upconv3, skip3), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)
        disp3up = upsample_nn_nearest(disp3)

        upconv2 = self.upconv2(iconv3)  # H/4
        concat2 = torch.cat((upconv2, skip2, disp3up), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)
        disp2up = upsample_nn_nearest(disp2)

        upconv1 = self.upconv1(iconv2)  # H/2
        concat1 = torch.cat((upconv1, skip1, disp2up), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)
        disp1up = upsample_nn_nearest(disp1)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, disp1up), 1)
        iconv0 = self.iconv0(concat0)
        disp0 = self.disp0(iconv0)

        scale_disps = [disp0 * 20, disp1 * 20, disp2 * 20, disp3 * 20]
        return scale_disps


class MonocularDistillLoss(nn.Module):
    def __init__(self):
        super(MonocularDistillLoss, self).__init__()

    def forward(self, disp_ests, stereo_disp_est):
        nscales = len(disp_ests)
        assert stereo_disp_est is not None

        stereo_disp_ests = generate_max_pyramid(stereo_disp_est)
        left_valid_mask = [(d > 0).float() for d in stereo_disp_ests]

        l1_loss = sum(
            [(disp_ests[i] / 20. - stereo_disp_ests[i] / 20.).abs()[left_valid_mask[i] > 0].mean() / (2 ** i) for i
             in range(nscales)])
        total_loss = l1_loss

        distill_disp_errors = [(disp_ests[i] - stereo_disp_ests[i]).abs() for i in range(nscales)]

        return total_loss, {"total_loss": total_loss, "l1_loss": l1_loss}, \
               {"distill_disp_errors": distill_disp_errors,
                "stereo_disp_ests": stereo_disp_ests}


class MonocularModel(nn.Module):
    def __init__(self):
        super(MonocularModel, self).__init__()
        self.model = MonocularVGG16(use_pretrained_weights=True)
        self.model_loss = MonocularDistillLoss()
        self.enable_flip_aug = True

    def forward(self, sample):
        # model
        left, right = sample["left"], sample["right"]
        do_flip_aug = self.training and self.enable_flip_aug and np.random.rand() > 0.5
        if do_flip_aug:
            left_disp_ests = [fliplr(d) for d in self.model(fliplr(left))]
        else:
            left_disp_ests = self.model(left)

        # model loss
        scalar_outputs, image_outputs = {}, {}
        if self.training:
            # train the network with estimated disparity from stereo network
            left_stereo_disp_est = sample["left_stereo_disp_est"]
            loss, scalar_outputs, image_outputs = self.model_loss(left_disp_ests, left_stereo_disp_est)

        # other outputs
        image_outputs["left"] = generate_pyramid(left)
        image_outputs["right"] = generate_pyramid(right)
        image_outputs["left_disp_est"] = left_disp_ests

        if "left_disp_gt" in sample:  # if ground truth is provided
            left_disp_gt = sample["left_disp_gt"]
            left_disp_gt = generate_max_pyramid(left_disp_gt)
            image_outputs["left_disp_gt"] = left_disp_gt
            image_outputs["left_disp_errors"] = [(est - gt).abs() * (gt > 0).float() for est, gt in zip(left_disp_ests, left_disp_gt)]

        # return outputs
        if self.training:
            return unsqueeze_dim0_tensor(loss), unsqueeze_dim0_tensor(scalar_outputs), image_outputs
        else:
            return left_disp_ests[0], unsqueeze_dim0_tensor(scalar_outputs), image_outputs

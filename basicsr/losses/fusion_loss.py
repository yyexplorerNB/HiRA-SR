import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY



@LOSS_REGISTRY.register()
class MaskFusionL1loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MaskFusionL1loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,image_vis, image_ir, generate_img, mask_vi):
        image_y = image_vis[:, :1, :, :]
        generate_img_y = generate_img[:, :1, :, :]

        mask_ir = torch.ones_like(mask_vi)
        mask_ir[mask_vi == 1] = 0

        loss_l1 = F.l1_loss(mask_vi * image_y, generate_img_y) + F.l1_loss(generate_img_y, mask_ir * image_ir)

        return self.loss_weight * loss_l1


@LOSS_REGISTRY.register()
class MaxFusionloss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MaxFusionloss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        generate_img_y = generate_img[:, :1, :, :]

        return self.loss_weight * F.l1_loss(generate_img_y, torch.max(image_y, image_ir))


@LOSS_REGISTRY.register()
class GradientFusionloss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientFusionloss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def gradient(self, input):
        """
        求图像梯度, sobel算子
        :param input:
        :return:
        """

        filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
        filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
        filter1.weight.data = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).reshape(1, 1, 3, 3).cuda()
        filter2.weight.data = torch.tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ]).reshape(1, 1, 3, 3).cuda()

        g1 = filter1(input)
        g2 = filter2(input)
        image_gradient = torch.abs(g1) + torch.abs(g2)

        return image_gradient

    def forward(self,image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        generate_img_y = generate_img[:, :1, :, :]

        gradient_generated_y = self.gradient(generate_img_y)
        gradient_image_y = self.gradient(image_y)
        gradient_ir = self.gradient(image_ir)

        return self.loss_weight * F.l1_loss(gradient_generated_y, torch.max(gradient_image_y, gradient_ir))
    
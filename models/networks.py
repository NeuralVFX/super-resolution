from torch.utils.data import *
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


############################################################################
# Re-usable blocks
############################################################################


def conv_block(ni, nf, kernel_size=3, relu=False):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if relu:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ConvRes(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(*[conv_block(nf, nf, relu=True), conv_block(nf, nf)])

    def forward(self, x):
        return x + self.block(x) * .1


def up_res(nf):
    return nn.Sequential(*[conv_block(nf, nf * 4), nn.PixelShuffle(2)])


############################################################################
# Generator
############################################################################


class SrResNet(nn.Module):
    # Generator with which RNN for up-sampling
    def __init__(self, nf, upres_count, switch_epoch=5):
        super().__init__()
        self.upres_count = upres_count
        features = [conv_block(3, nf)]
        [features.append(ConvRes(nf)) for i in range(8)]
        features += [conv_block(nf, nf)]
        self.res = nn.Sequential(*features)
        self.upsample = nn.ModuleList([up_res(nf) for a in range(self.upres_count)])
        self.out = nn.Sequential(*[nn.BatchNorm2d(nf), conv_block(nf, 3)])
        self.rnn = True
        self.first_run = True
        self.switch_epoch = switch_epoch

    def forward(self, x, epoch=0):
        self.check_rnn_status(epoch)

        x = self.res(x)
        if self.rnn:
            for i in range(self.upres_count):
                x = self.upsample[0](x)
        else:
            for i in range(self.upres_count):
                x = self.upsample[i](x)
        x = self.out(x)
        return x

    def check_rnn_status(self, epoch):
        pass
        # check if RNN must be on
        """
        if self.first_run:
            if epoch > self.switch_epoch:
                print("Switching RNN Off")
                self.rnn = False
            self.first_run = False
        # switch RNN of when we hit switch epoch
        if self.rnn:
            if epoch == self.switch_epoch:
                self.rnn_switch(False)
        """

    def rnn_switch(self, rnn_on):
        # Switch which turns off RNN, and copies trained weights into separate trainable layers for fine tuning
        if rnn_on:
            self.rnn = True
        else:
            self.rnn = False
            print("Switching RNN Off, Copying Weights")
            for i in range(self.upres_count):
                weight = self.upsample[0][0][0].weight.clone()
                self.upsample[i][0][0].weight = nn.Parameter(weight)


def make_vgg():
    vgg = models.vgg16(pretrained=True)
    children = list(vgg.children())
    children.pop()
    vgg = children[0]
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


############################################################################
# Hook and Losses
############################################################################


class SetHook:
    # Register hook inside of network to retrieve features
    feats = None

    def __init__(self, block):
        self.hook_reg = block.register_forward_hook(self.hook)

    def hook(self, module, hook_input, output):
        self.feats = output

    def close(self):
        self.hook_reg.remove()


class ContLoss(nn.Module):
    # Store Hook, Calculate Content Loss
    def __init__(self, vgg, ct_wgt, l1_weight, content_layer_ids):
        super().__init__()
        self.m, self.ct_wgt, self.l1_weight = vgg, ct_wgt, l1_weight
        self.cfs = [SetHook(vgg[i]) for i in content_layer_ids]

    def forward(self, input_img, target_img):
        self.m(target_img.data)
        result_l1 = [F.l1_loss(input_img, target_img)*self.l1_weight]
        targ_feats = [o.feats.data.clone() for o in self.cfs]
        self.m(input_img)
        inp_feats = [o.feats for o in self.cfs]
        result_ct = [F.l1_loss(inp.view(-1), targ.view(-1)) * self.ct_wgt for inp, targ in zip(inp_feats, targ_feats)]
        return result_ct, result_l1

    def close(self):
        [o.remove() for o in self.sfs]

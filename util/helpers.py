from torch.utils.data import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


############################################################################
# Helper Utilities
############################################################################


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    # initiate pre-pixel shuffle conv layers with ICNR
    new_shape = [int(x.shape[0])/(scale**2)] + list(x.shape[1:])
    single_kernel = torch.zeros(new_shape)
    single_kernel = init(single_kernel)
    single_kernel = single_kernel.transpose(0, 1)
    single_kernel = single_kernel.contiguous().view(single_kernel.shape[0], single_kernel.shape[1], -1)
    full_kernel = single_kernel.repeat(1, 1, scale**2)
    transposed_shape = [x.shape[1]]+[x.shape[0]]+list(x.shape[2:])
    full_kernel = full_kernel.contiguous().view(transposed_shape)
    full_kernel = full_kernel.transpose(0, 1)
    return full_kernel


class WeightsInit:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, m):
        # Set initial state of weights
        classname = m.__class__.__name__
        if 'ConvTrans' == classname:
            pass
        elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
            nn.init.normal_(m.weight.data, 0, .02)

        if classname == 'Conv2d':
            if m.out_channels == self.channels:
                kern = icnr(m.weight)
                m.weight.data.copy_(kern)
                print(f'Init with ICNR:{classname}')


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


############################################################################
# Display Images
############################################################################


def show_test(params, denorm, dataloader, model, save=False):
    # Show and save
    ids_a = params['ids']
    image_grid_len = len(ids_a)
    fig, ax = plt.subplots(image_grid_len, 3, figsize=(13, 4.5 * image_grid_len))
    count = 0
    model.eval()
    for idx, real in enumerate(dataloader):
        if idx in ids_a:
            low_res = Variable(real[0].cuda())
            high_res = Variable(real[1].cuda())
            test = model(low_res)
            ax[count, 0].cla()
            ax[count, 0].imshow(denorm.denorm(low_res[0]))
            ax[count, 1].cla()
            ax[count, 1].imshow(denorm.denorm(test[0]))
            ax[count, 2].cla()
            ax[count, 2].imshow(denorm.denorm(high_res[0]))
            count += 1
    model.train()
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)


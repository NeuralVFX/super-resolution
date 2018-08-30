#!/usr/bin/env python
import argparse
from super_resolution import *


parser = argparse.ArgumentParser()


parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='imagenet', type=str)
parser.add_argument('--batch_size', nargs='?', default=1, type=int)
parser.add_argument('--gen_filters', nargs='?', default=64, type=int)
parser.add_argument('--blur_kernel', nargs='?', default=15, type=int)
parser.add_argument('--lr_drop_every', nargs='?', default=5, type=int)
parser.add_argument('--lr_drop_start', nargs='?', default=5, type=int)
parser.add_argument('--rnn_switch_epoch', nargs='?', default=4, type=int)
parser.add_argument('--vgg_layers_c', type=int, nargs='+', default=[3, 8, 15])
parser.add_argument('--zoom_count', nargs='?', default=3, type=int)
parser.add_argument('--l1_weight', nargs='?', default=.1, type=float)
parser.add_argument('--content_weight', nargs='?', default=.5, type=float)
parser.add_argument('--in_res', nargs='?', default=64, type=int)
parser.add_argument('--lr', nargs='?', default=1e-3, type=float)
parser.add_argument('--train_epoch', nargs='?', default=60, type=int)
parser.add_argument('--test_perc', nargs='?', default=.1, type=float)
parser.add_argument('--data_perc', nargs='?', default=1, type=float)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--workers', nargs='?', default=4, type=int)
parser.add_argument('--save_every', nargs='?', default=5, type=int)
parser.add_argument('--save_img_every', nargs='?', default=1, type=int)
parser.add_argument('--ids', type=int, nargs='+', default=[1,2,3,4])
parser.add_argument('--save_root', nargs='?', default='imagenet_upres', type=str)
parser.add_argument('--load_state', nargs='?', type=str)


params = vars(parser.parse_args())

# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    sr = SuperResolution(params)
    if params['load_state']:
        sr.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    sr.train()

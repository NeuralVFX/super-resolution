import math
import copy
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import *
import matplotlib.pyplot as plt

from util import helpers as helper
from util import loaders as load
from models import networks as n

plt.switch_backend('agg')


############################################################################
# Train
############################################################################


class SuperResolution:
    """
    Example usage if not using command line:

    from super_resolution import *
    params = {'dataset': 'imagenet',
                  'batch_size':1,
                  'workers':4,
                  'vgg_layers_c':[3,8,15],
                  'lr':1e-4,
                  'beta1': .5,
                  'beta2': .999,
                  'gen_filters': 64,
                  'content_weight': .5,
                  'l1_weight':  .1,
                  'test_perc': .01,
                  'data_perc': 1,
                  'zoom_count': 3,
                  'train_epoch': 100,
                  'in_res':64,
                  'save_every': 1,
                  'ids': [16, 26],
                  'rnn_switch_epoch':4,
                  'save_img_every':1,
                  'lr_drop_start': 5,
                  'lr_drop_every': 5,
                  'blur_kernel':15,
                  'save_root': 'super_res_imagenet'}
    sr = SuperResolution(params)

    sr.train(params)
    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0

        # Setup data loaders
        self.transform = load.NormDenorm([.485, .456, .406], [.229, .224, .225])

        self.train_data = load.SuperResDataset(f'/data/{params["dataset"]}/train',
                                               self.transform,
                                               in_res=params["in_res"],
                                               out_res=params["in_res"] * math.pow(2, params["zoom_count"]),
                                               data_perc=params["data_perc"],
                                               kernel=15)

        self.test_data = copy.deepcopy(self.train_data)
        self.test_data.train = False

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=params["batch_size"],
                                                        num_workers=params["workers"],
                                                        shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=params["batch_size"],
                                                       num_workers=params["workers"],
                                                       shuffle=True, drop_last=True)

        self.data_len = self.train_data.__len__()
        self.test_data = copy.deepcopy(self.train_data)
        self.test_data.train = False
        print(f'Data Loaders Initialized,  Data Len:{self.train_data.__len__()}')

        # Setup models
        self.model_dict['G'] = n.SrResNet(params["gen_filters"],
                                          params["zoom_count"],
                                          switch_epoch=params["rnn_switch_epoch"])

        self.weights_init = helper.WeightsInit(params["gen_filters"]*4)
        for i in self.model_dict.keys():
            self.model_dict[i].apply(self.weights_init)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')
        # Setup loss

        self.vgg = n.make_vgg()
        self.vgg.cuda()

        self.ct_loss = n.ContLoss(self.vgg,
                                  params['content_weight'],
                                  params['l1_weight'],
                                  params['vgg_layers_c'])

        self.ct_loss.cuda()

        # Setup optimizers
        self.opt_dict["G"] = optim.Adam(self.model_dict["G"].parameters(),
                                        lr=params['lr'],
                                        betas=(params['beta1'],
                                               params['beta2']))

        print('Losses Initialized')

        # Setup history storage
        self.losses = ['L1_Loss', 'C_Loss']
        self.loss_batch_dict = {}
        self.loss_batch_dict_test = {}
        self.loss_epoch_dict = {}
        self.loss_epoch_dict_test = {}
        self.train_hist_dict = {}
        self.train_hist_dict_test = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []
            self.train_hist_dict_test[loss] = []
            self.loss_epoch_dict_test[loss] = []
            self.loss_batch_dict_test[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1

        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])

        self.train_hist_dict = state['train_hist']
        self.train_hist_dict_test = state['train_hist_test']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict,
                       'train_hist_test': self.train_hist_dict_test
                       }

        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            x_test = range(len(self.train_hist_dict_test[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
                plt.plot(x_test, self.train_hist_dict_test[key], label=key + '_test')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def lr_lookup(self):
        # Determine proper learning rate multiplier for this iter, cuts in half every "lr_drop_every"
        div = max(0, ((self.current_epoch - self.params["lr_drop_start"]) // self.params["lr_drop_every"]))
        lr_mult = 1 / math.pow(2, div)
        return lr_mult

    def train_gen(self, low_res, high_res):
        # train function for generator
        self.opt_dict["G"].zero_grad()

        high_fake = self.model_dict["G"](low_res, self.current_epoch)
        ct_losses, l1_losses = self.ct_loss(high_fake, high_res)

        self.loss_batch_dict['L1_Loss'] = sum(l1_losses) / self.params['batch_size']
        self.loss_batch_dict['C_Loss'] = sum(ct_losses) / self.params['batch_size']

        # Step
        total_loss = self.loss_batch_dict['L1_Loss'] + self.loss_batch_dict['C_Loss']
        total_loss.backward()
        self.opt_dict["G"].step()
        return l1_losses, ct_losses

    def test_gen(self, low_res, high_res):
        # test function for generator
        high_fake = self.model_dict["G"](low_res, self.current_epoch)
        ct_losses, l1_losses = self.ct_loss(high_fake, high_res)

        self.loss_batch_dict_test['L1_Loss'] = sum(l1_losses) / self.params['batch_size']
        self.loss_batch_dict_test['C_Loss'] = sum(ct_losses) / self.params['batch_size']
        return l1_losses, ct_losses

    def test_loop(self):
        # Test on validation set
        self.model_dict["G"].eval()

        for loss in self.losses:
            self.loss_epoch_dict_test[loss] = []
        # test loop #
        for low_res, high_res in tqdm(self.test_loader):
            low_res = Variable(low_res.cuda())
            high_res = Variable(high_res.cuda())

            # TEST GENERATOR
            l1_losses, content_losses = self.test_gen(low_res, high_res)

            # append all losses in loss dict #
            [self.loss_epoch_dict_test[loss].append(self.loss_batch_dict_test[loss].item()) for loss in self.losses]
        [self.train_hist_dict_test[loss].append(helper.mft(self.loss_epoch_dict_test[loss])) for loss in self.losses]

    def train_loop(self):
        # Train on train set
        self.model_dict["G"].train()

        for loss in self.losses:
            self.loss_epoch_dict[loss] = []

        lr_mult = self.lr_lookup()
        self.opt_dict["G"].param_groups[0]['lr'] = lr_mult * self.params['lr']
        print(f"Sched Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
        [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in self.opt_dict.keys()]
        for low_res, high_res in tqdm(self.train_loader):
            low_res = Variable(low_res.cuda())
            high_res = Variable(high_res.cuda())

            # TRAIN GENERATOR
            l1_losses, content_losses = self.train_gen(low_res, high_res)

            # append all losses in loss dict
            [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].item()) for loss in self.losses]
            self.current_iter += 1
        [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]


    def train(self):
        # Train following learning rate schedule
        params = self.params
        while self.current_epoch < params["train_epoch"]:
            epoch_start_time = time.time()

            # TRAIN LOOP
            self.train_loop()

            # TEST LOOP
            self.test_loop()

            # generate test images and save to disk
            if self.current_epoch % params["save_img_every"] == 0:
                helper.show_test(params,
                                 self.transform,
                                 self.test_loader,
                                 self.model_dict['G'],
                                 save=f'output/{params["save_root"]}_val_{self.current_epoch}.jpg')
            # save
            if self.current_epoch % params["save_every"] == 0:
                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                tqdm.write(save_str)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(f'Epoch Training Training Time: {per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
            [print(f'Val {loss}: {helper.mft(self.loss_epoch_dict_test[loss])}') for loss in self.losses]
            print('\n')
            self.current_epoch += 1

        self.display_history()
        print('Hit End of Learning Schedule!')

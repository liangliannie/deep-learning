#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
import options
import os
from pytorch_msssim import MSSSIM
from MedGan import MedGanloss

import Unnet

        
class Sino_repair_net():
    
    def __init__(self, opts, device_ids, load_model=False):
        self.opts = opts
        self.epoch_step = 0
        self.model_num = 0
        self.network = Unnet.UNet(opts)
        # self.network = Resnet.resnet152()
        # self.network = framelet.Framelets()
        # self.network = googlenet.GoogLeNet()
        # self.network = densenet.densenet161()
        
        if torch.cuda.device_count() > 1 and opts.max_gpus > 1:
            if len(device_ids) <= opts.max_gpus:
                self.network = torch.nn.DataParallel(self.network) #, device_ids=device_ids[0]
            else:
                self.network = torch.nn.DataParallel(self.network, device_ids=device_ids[0:opts.max_gpus-1])
        self.network.cuda()
        
        # Create two sets of loss functions
        self.loss_func_l1 = torch.nn.L1Loss()
        self.loss_func_MSE = torch.nn.MSELoss()
        self.MedGanloss = MedGanloss()
        self.mssim_loss = MSSSIM(window_size=9, size_average=True)
        self.loss_func_poss = torch.nn.PoissonNLLLoss()
        self.loss_func_KLDiv = torch.nn.KLDivLoss()
        self.loss_func_Smoothl1 = torch.nn.SmoothL1Loss()
        self.loss_func_part = torch.nn.L1Loss()
        self.test_loss = torch.nn.MSELoss(reduction='none')
        self.averagepool = torch.nn.AvgPool2d(3, stride=2)
        self.optim_count = 0
        #TODO2: Change the load model dict
        if self.opts.load_model == True or load_model:
            print("Restoring model")
            try:
                if load_model:
                    self.network.load_state_dict(torch.load('/home/liang/Desktop/output/model/model_dict_0'))

                else:
                    self.network.load_state_dict(torch.load(os.path.join(self.opts.output_path, 'model', 'model_dict')))
            except:
                # original saved file with DataParallel
                state_dict = torch.load(os.path.join(self.opts.output_path, 'model', 'model_dict'))
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # name = k[7:] # remove `module.`
                    name = 'module.'+ k
                    new_state_dict[name] = v
                # load params
                self.network.load_state_dict(new_state_dict)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_batch(self, input_img, target_img, valid=None):

        if valid is None:
            output = self.network.forward(input_img)
            loss, loss2 = self.optimize(output, target_img)
            return output, loss, loss2
        else:

            final = input_img.clone()
            mask = torch.tensor([i for i, n in enumerate(valid) if n==1]).cuda()
            if len(mask) > 0:
                traininput = torch.index_select(input_img, 0, mask)
                trainoutput = self.network.forward(traininput)
                loss, loss2 = self.optimize(trainoutput, torch.index_select(target_img, 0, mask))
                final[mask] = trainoutput
            else:
                loss, loss2 = self.loss_func_l1(final, target_img), (1 - self.mssim_loss.forward(final, target_img))/2

            return final, loss, loss2

    def test(self, x, y, valid=None):

        if valid is None:
            output = self.network.forward(x)
            loss = self.test_loss(output, y).detach()
            return output, loss
        else:
            final = x.clone()
            mask = torch.tensor([i for i, n in enumerate(valid) if n == 1]).cuda()
            if len(mask) > 0:
                traininput = torch.index_select(x, 0, mask)
                trainoutput = self.network.forward(traininput)
                final[mask] = trainoutput
            loss = self.test_loss(final, y).detach()
        return final, loss

    def optimize(self, output, target_img):
        #TODO: can add other loss terms if needed
        #TODO: need to step though this code to make sure it works correctly
        input1 = output#torch.floor(output)  #/ (output.max() + 1e-8)
        input2 = target_img #/ (target_img.max() + 1e-8)
        # # Including l1 loss
        # mask = ((input_img * -1.0) + 1.0) >= 0.8
        loss1 = self.loss_func_l1(input1, input2)
        l1 = loss1.detach()

        # Including a consistency loss
        loss2 = self.loss_func_MSE(input1, input2)
        l2 = loss2.detach()

        loss3 = 0.0001*(1 - self.mssim_loss.forward(input1, input2))
        l3= loss3.detach()
        #
        # loss3 = self.MedGanloss(output, target_img)
        # l3 = loss3.detach()
        # loss3 = self.loss_func_l1(self.averagepool(input1), self.averagepool(input2))
        # loss3 = abs(torch.floor(input1).mean()- input2.mean())

        # loss3 = self.loss_func_MSE(output, target_img)
        #
        # if self.OPT_count == 0:
        #     self.alpha = torch.tensor(0.5).cuda()
        #     self.lossl1 = []
        #     self.lossmssim = []
        #
        # self.lossl1.append(l1.item())
        # self.lossmssim.append(l2.item())
        #
        # if self.OPT_count >= 20:
        #     self.alpha = torch.FloatTensor(self.lossl1).mean().cuda()/(torch.FloatTensor(self.lossl1).mean() + torch.FloatTensor(self.lossmssim).mean()).cuda()
        #     self.OPT_count = 0
        #
        # # loss = self.alpha * loss1+(1-self.alpha) *loss2
        # vx = output - torch.mean(output)
        # vy = target_img - torch.mean(target_img)
        # loss_pearson_correlation = 1 - torch.sum(vx * vy) / (
        #             torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2)))  # use Pearson correlation


        loss = loss1 + loss2 + loss3
        # print(loss1, loss2, loss3)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optim_count += 1

        return l1, l2

    def save_network(self):
        print("saving network parameters")
        folder_path = os.path.join(self.opts.output_path, 'model')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.network.state_dict(), os.path.join(folder_path, "model_dict_{}".format(self.model_num)))
        self.model_num += 1
        if self.model_num >= 5: self.model_num = 0


if __name__ == '__main__':
    opts = options.parse()
    # net = UNet(opts).cuda()
    # Change loading module for single test
    net = Sino_repair_net(opts, [0, 1], load_model=True)
    net.cuda()
    # print(net)

    import matplotlib.pylab as plt
    import numpy as np


    def plot_img(img):
        # Pass in the index to read one of the sinogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Original (Sinogram)")
        ax.set_xlabel("Projection position (pixels)")
        ax.set_ylabel("Projection angle (deg)")
        img = np.transpose(img)
        ax.imshow(img)

    from datasets import Sino_Dataset
    # data = Sino_Dataset('/home/liang/Desktop/dataset.pkl', 5, testing=False, input_depth=3)
    data = Sino_Dataset(opts.test_file, 5, testing=True, input_depth=opts.input_channel_number, output_depth=opts.output_channel_number)

    output = []
    for i, (good, bad) in enumerate(data):
        # out = good
        # if i == 0:
        #     for j in range(len(out)):
        #         output.append(out[j])
        # else:
        #     output.append(out[-1])
            # print(i)
        # print(good.shape)
        # bad = bad[0]
        # good = good[0]

        good /= (good.max() + 1e-8)
        bad /= (bad.max() + 1e-8)

        test_x = torch.from_numpy(bad)
        test_x = test_x.unsqueeze(0)
        # # test_x = test_x.unsqueeze(0)
        test_x = Variable(test_x).cuda()
        # #
        test_y = torch.from_numpy(good)
        test_y = test_y.unsqueeze(0)
        # # test_y = test_y.unsqueeze(0)
        test_y = Variable(test_y).cuda()

        # out_x = net.network.forward(test_x)
        # out_x, loss = net.test(test_y, test_x)
        # out_x = out_x.squeeze()
        # out_x = out_x.cpu()
        # const = out_x.detach().numpy()
        # plot_img(good* 255)
        # plot_img(bad* 255)
        # plot_img(const* 255)
        # # print(loss)
        #
        # break
    output = np.stack(output)
    print(output.shape)
    plt.show()


    #
    # test_x = Variable(torch.FloatTensor(1, 1, 64, 520)).cuda()
    # out_x = net.forward(test_x)



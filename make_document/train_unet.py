#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 18:45:25 2019

@author: bill
@updated: Liang
"""
# import memory_profiler
import torch

import numpy as np
import os, subprocess, time
import options, datasets
import unet
from scipy import ndimage
import scipy.ndimage
import scipy.misc
from random import random
# import sino_repair_utils as sr_utils
from PIL import Image
# from sklearn.preprocessing import RobustScaler as scale
# from sklearn.preprocessing import MinMaxScaler as scale
# from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                          denoise_wavelet)
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.preprocessing import StandardScaler

try:
    import visualize
except ImportError:
    pass

def preprocess(input_img, target_img, train=False):

    Scale_Means = []
    for idx in range(len(input_img)):
        # shape = input_img[idx].shape
        # scaler = StandardScaler(with_mean=False)
        # data = input_img[idx][0].cpu().numpy()
        # scaler.fit(data)
        # input_img[idx] = torch.tensor(data).reshape(shape).cuda()
        mean_input = input_img[idx].mean()
        max_input = input_img[idx].std()+1e-8
        input_img[idx] = (input_img[idx] - mean_input) / max_input
        Scale_Means.append((mean_input, max_input))
    return Scale_Means, input_img, target_img


def fill_image(input_sino, out_img):
    # resample the output as poisson
    # print(input_sino.shape)
    test = out_img.copy()
    # test[0] = scipy.ndimage.generic_filter(test[0], lambda x: np.random.poisson(x), size=(1,1))
    # scipy.ndimasge.generic_filter(test, np.random.poisson, size=(1, 1))
    test =  np.random.poisson(test)
    mask = input_sino * (-1.0) + 1.0
    mask = mask > 0
    # mask = input_sino <= test
    # scale = (input_sino*(1-mask)).mean()/np.ceil(test*mask).mean()
    final = np.ceil(test*mask) + input_sino
    # final = test

    return final


def main():
    # Parse the command line arguments
    opts = options.parse()
    # torch.backends.cudnn.benchmark = False
    device_ids = [0, 1]
    # if opts.scan_gpus == True:
    #     # Scan the available GPUs to find one currently using less than 10Mb
    #     device_ids = sr_utils.wait_for_GPU(10)
    #     available_gpus = str(device_ids)[1:-1]
    #     print("Using GPUs {}".format(available_gpus))
    #     os.environ['CUDA_VISIBLE_DEVICES']=str(available_gpus)
    #
    #
    # # If using synthetic data launch the subprocess that keeps creating sinograms
    # if opts.synthetic_data == True:
    #     print("Launching the sinogram creation sub process")
    #     p = subprocess.Popen(["python", "create_sinos.py"])
    #     while(os.path.isfile("training_data.npy") == False):
    #         print("waiting for training data...")
    #         time.sleep(5)
        

    # Create the Network trainer and train!
    # device_ids = None
    trainer = Network_Trainer(opts, device_ids)
    trainer.train()

    # Clean up the subprocess when generating synthetic data, but if you kill the training using ctl-C the process will be orphaned
    # and you'll have to manually clean up with something on the command line like >pkill python
    # if opts.synthetic_data == True:
    #     p.kill()


class Network_Trainer():
    # @profile
    def __init__(self, opts, device_ids):

        if opts.visualize:
            print("Initializing training visualization\n")
            self.viz = visualize.Visualize_Training()
        
        self.log_path = os.path.join(opts.output_path, "log")
        if not os.path.exists(os.path.join(self.log_path, "test")):
            os.makedirs(os.path.join(self.log_path, "test"))
            
        self.opts = opts
        self.device_ids = device_ids
        self.test_data = None
        self.scaler = 5
        self.batchcount = 0
        self.loss_func = torch.nn.L1Loss()

    # @profile
    def train(self):    
    

        # Create the neural network
        network = unet.Sino_repair_net(self.opts, self.device_ids)

        self.training_data = datasets.Sino_Dataset(self.opts.training_file, self.opts.epoch_size, testing=False, loading_multiplefiles=self.opts.loading_multiplefiles, input_depth=self.opts.input_channel_number, output_depth=self.opts.output_channel_number)
        train_loader = torch.utils.data.DataLoader(self.training_data, batch_size=self.opts.batch_size, shuffle=True, drop_last=True)
        # Set the parameters of the warm reset
        next_reset = self.opts.warm_reset_length
        warm_reset_increment = 0
        
        for epoch in range(self.opts.epochs):
            print("Starting epoch: " + str(epoch)+'\n')
            
            if epoch % next_reset == 0:
                print("Resetting Optimizer\n")
                optimizer = torch.optim.Adam(network.network.parameters(), lr=self.opts.initial_lr, betas=(0.5, 0.999))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.opts.lr_decay)
                network.set_optimizer(optimizer)

                # Set the next reset
                next_reset += self.opts.warm_reset_length + warm_reset_increment
                warm_reset_increment += self.opts.warm_reset_increment
            else:
                scheduler.step()
        
            for param_group in optimizer.param_groups:
                print("Current learning rate: {}\n".format(param_group['lr']))

            network.network.train() # start Training Mode

            for steps, (orig_sino, bad_sino, valid) in enumerate(train_loader): #load the batch size 10*1*64*520
                # Create the target tensor (y)

                input_img = bad_sino.cuda()
                target_img = orig_sino.cuda()
                valid = valid.cuda()

                Scale_Means, input_img, target_img = preprocess(input_img, target_img, train=True)

                input_img, target_img = input_img.type(torch.float32), target_img.type(torch.float32)
                output, loss, loss2 = network.train_batch(input_img, target_img, valid)

                print('l1 loss: {:.5e}, consistency loss: {:.5e}\n'.format(loss, loss2), flush=True)

                # for idx in range(self.opts.batch_size):
                #     input_img[idx] = input_img[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]
                #     target_img[idx] = target_img[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]
                #     output[idx] = output[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]


                if steps % 2 == 0 and self.opts.visualize:
                    self.visualize_progress(self.opts,
                                    [target_img.clone().detach(), input_img.clone().detach(), output.clone().detach()],
                                    istest=False)

            # Done with a training epoch
            del target_img, input_img, output

            # Save the network
            if epoch%21 == 20:
                network.save_network()
            
            # Run through a test set
            if self.opts.check_test:
                self.test(network, self.opts.test_batch_size)

            # Load the next data file
            if self.opts.synthetic_data:
                try: 
                    self.training_data.loadFile(remove=True)
                except:
                    time.sleep(10)
                    self.training_data.loadFile(remove=True)

        print("Done\n")

    # @profile
    def test(self, network, batch_size, perform_recon_loss=False, output_all_sinos=False):

        print("\nRunning test data")
    
        # Initialize the data loader
        if not self.test_data:
            self.test_data = datasets.Sino_Dataset(self.opts.test_file, self.opts.test_set_size, testing=True, loading_multiplefiles=self.opts.loading_multiplefiles, input_depth=self.opts.input_channel_number, output_depth=self.opts.output_channel_number, is_test_in_train=True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=False, drop_last=True) 
        
        test_loss = 0
        network.network.eval() # sets the module in evaluation mode
        # Iterate through the test set
        batch_count = 0

        for steps, (orig_sino, bad_sino, valid) in enumerate(test_loader):  # load the batch size 10*1*64*520

            input_img = bad_sino.clone().detach().cuda()
            target_img = orig_sino.clone().detach().cuda()
            valid = valid.cuda()

            Scale_Means, input_img, target_img = preprocess(input_img, target_img, train=True)

            input_img, target_img = input_img.type(torch.float32), target_img.type(torch.float32)
            output, loss, loss2 = network.test(input_img, target_img, valid)

            print('l1 loss: {:.5e}, consistency loss: {:.5e}\n'.format(loss, loss2), flush=True)

            for idx in range(self.opts.batch_size):
                input_img[idx] = input_img[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]
                target_img[idx] = target_img[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]
                output[idx] = output[idx] * Scale_Means[idx][1] + Scale_Means[idx][0]

            test_loss += loss.sum()

            if self.opts.visualize:

                self.visualize_progress(self.opts,
                                        [target_img.clone().detach(), input_img.clone().detach(), output.clone().detach()],
                                        istest=True)
            
            del target_img, input_img, output  # This is questionable
            
            batch_count += 1
            self.batchcount += 1
            self.batchcount %= 10
        
        print("Total testing loss: {:.5f}\n".format(test_loss))
        test_log_path = self.write_log(test_loss.item(), log_type="test")

        if self.opts.visualize:
            self.viz.Plot_Progress(test_log_path, "test")

    # @profile
    def write_log(self, value, log_type="test"):
        
        # If the log file doesn't exist create it, then record the current loss for that batch
        log_path = os.path.join(self.opts.output_path, "log")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        if log_type == "test":
            log_path = os.path.join(log_path, "test_log.txt")
        elif log_type == "test_recon":
            log_path = os.path.join(log_path, "recon_log.txt")
        elif log_type == "train_loss":
            log_path = os.path.join(log_path, "train_log.txt")
        elif log_type == "train_loss2":
            log_path = os.path.join(log_path, "train_log2.txt")
        
        with open(log_path, 'a') as file:
            file.write(str(value)+'*' + '\n')
            
        return log_path

    # @profile
    def visualize_progress(self, opts, images, istest=False):
        # This function logs training progress and optionally visualizes the progress
    
        target_img, input_img, output = images

        # Make the images suitable for display and pull them off the GPU
        target_img  = target_img.data.cpu().numpy()[0]
        output      = output.data.cpu().numpy()[0]
        input_img   = input_img.data.cpu().numpy()[0]
        input_img -= input_img.min()
        final_img = output #.astype(np.uint16)
        # final_img   = fill_image(input_img, output)

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        error = np.abs(output-target_img)
        print('Test Loss per Pixel:', error.mean(),'\n')

        # # Scale the images to be 8-bit integers
        output = output/(output.max()+ 1e-8)*255
        target_img = target_img/(target_img.max() + 1e-8)*255
        input_sino = input_img/(input_img.max()+ 1e-8)*255
        final_img = final_img/(final_img.max()+ 1e-8)*255
        diff_img = np.absolute(target_img - final_img)

        # Visualize training samples

        if opts.output_channel_number <= 3:
            showchannel = opts.output_channel_number
        else:
            showchannel = 3
        if output.shape[1] <= 64:
            step = 1
        else:
            step = 2
        images = np.stack((target_img[:showchannel][:, ::step, ::step], input_sino[:showchannel][:, ::step, ::step], output[:showchannel][:, ::step, ::step], final_img[:showchannel][:, ::step,::step],
                           diff_img[:showchannel][:, ::step, ::step]), axis=0)

        if not istest:
            self.viz.Show_Train_Images(images, text='Train: Target  + Input + Output + Final +  Diff + mask')
        else:
            self.viz.Show_Test_Images(images, text='Test: Target  + Input + Output + Final + Diff + mask')
        
        # Save copies of the training samples
        good_img_path   = os.path.join(self.log_path, "orig.png")
        bad_img_path    = os.path.join(self.log_path, "corrupt.png")
        result_img_path = os.path.join(self.log_path, "result.png")
        diff_img_path   = os.path.join(self.log_path, "diff.png")
        out_img_path    = os.path.join(self.log_path, "out.png")

        try:
            Image.fromarray(np.squeeze(target_img).astype(np.uint8), mode='L').save(good_img_path)
            Image.fromarray(np.squeeze(input_sino).astype(np.uint8), mode='L').save(bad_img_path)
            Image.fromarray(np.squeeze(final_img).astype(np.uint8), mode='L').save(result_img_path)
            Image.fromarray(np.squeeze(diff_img).astype(np.uint8), mode='L').save(diff_img_path)
            Image.fromarray(np.squeeze(output).astype(np.uint8), mode='L').save(out_img_path)
        except:
            pass


if __name__ == "__main__":
    main()

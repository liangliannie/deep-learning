#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
import random, pickle
import glob
import os
import cv2 as cv

class Sino_Dataset(Dataset):
    '''
     Sinogram dataset
    '''

    # __slots__ = ['data_file', 'epoch_size', 'testing', 'max_missing', 'min_missing']
    def __init__(self, datafile,  epoch_size, testing=False, loading_multiplefiles=False, input_depth=1, output_depth=1, is_test_in_train=False):
        self.is_test = testing
        self.is_test_in_train = is_test_in_train

        self.loading_multiplefiles = loading_multiplefiles

        self.epoch_size = epoch_size
        self.input_depth = input_depth
        self.output_depth = output_depth

        self.previous = None
        self.DataQueryCount = None
        self.LoadNewFile = True
        self.trainfile = []
        self.trainfile_init = 0
        self.init = False
        self.do_feature_extract_classify = False

        if os.path.isdir(datafile):
            for file in sorted(glob.glob("{}*.pkl".format(datafile))):
                self.trainfile.append(file)
        else:
            self.trainfile.append(datafile)

    def loadFile(self, data_file="training_data.npy"):
        # print("Loading new data file")
        try:
            with open(data_file, 'rb') as file:
                self.data = pickle.load(file).astype('float32')
                # self.data = self.data[:, :int(self.data.shape[1]%100),:,:]
                # self.mask = np.ones(self.data.shape[1:]).astype('float32')

                self.datalen = self.data.shape[1]
                print(self.data.shape)

        except IOError:
            raise ValueError("Couldn't open the sinogram data file")

        # if self.is_test == False:
        #     self.status_list = np.ones(self.data.shape[0])  # see comment below

            # self.status = 0
    def setMaxMissing(self, max_missing):

        self.max_missing = max_missing


    def checkfeature(self, bad):
        # print(bad.shape)
        img = ((bad / (bad.max() + 1e-8)) * 255).astype(np.uint8)[0]
        img = cv.blur(img, (3, 3))
        edges = cv.Canny(img, 50, 150).reshape(1, bad.shape[-2], bad.shape[-1])
        if not self.do_feature_extract_classify:
            return 1
        return 1#int(edges.mean() >= .25)

    def __len__(self):

        return self.epoch_size

    def __getitem__(self, idx):


        if self.init and self.DataQueryCount >= (self.datalen - self.input_depth + 1):


            if self.is_test and not self.is_test_in_train:
                raise StopIteration


            if self.loading_multiplefiles:
                self.LoadNewFile = True
            else:
                self.LoadNewFile = False
                self.DataQueryCount = 0




        if self.LoadNewFile:

            if self.is_test:
                print('Loading test file ' + self.trainfile[self.trainfile_init])
            else:
                print('Loading train file ' + self.trainfile[self.trainfile_init])

            self.loadFile(self.trainfile[self.trainfile_init])
            self.trainfile_init += 1
            self.trainfile_init %= len(self.trainfile)
            self.LoadNewFile = False
            self.init = True

            self.idxlist = np.arange(self.datalen - self.input_depth + 1)
            np.random.shuffle(self.idxlist)
            self.DataQueryCount = 0

        # print(self.DataQueryCount, (self.datalen - self.input_depth + 1))

        if type(idx) != self.previous:

            if isinstance(idx, int) or self.input_depth >= 2:  # The depth can be choose based on input data type

                self.idxlist = np.arange(self.datalen - self.input_depth + 1)

            if isinstance(idx, slice):

                self.idxlist = np.arange(self.datalen-len(range(*idx.indices(10 ** 3)))+1)

            if not self.is_test:

                np.random.shuffle(self.idxlist)

            self.DataQueryCount = 0

            self.previous = type(idx)   # Consider multiple case input

        if isinstance(idx, int) or self.input_depth >= 2:

            idx = self.idxlist[self.DataQueryCount]

            if self.input_depth >= 2:

                corrupt_sino = self.data[1, idx:idx + self.input_depth]
                orig_sino = self.data[0, idx:idx + self.input_depth]
                # mask_sino = self.mask[idx:idx + self.input_depth]

            else:

                orig_sino = self.data[0, idx]  # X
                corrupt_sino = self.data[1, idx]
                # mask_sino = self.mask[idx]

                # Expand the dimensions to imag case
            orig_sino = np.expand_dims(orig_sino, axis=0)
            corrupt_sino = np.expand_dims(corrupt_sino, axis=0)
            # mask_sino = np.expand_dims(mask_sino, axis=0)

        if isinstance(idx, slice):

            idx = self.idxlist[self.DataQueryCount]

            orig_sino = self.data[0, idx:idx+self.channelnumber]
            corrupt_sino = self.data[1, idx:idx+self.channelnumber]

        # orig_sino = np.fft.fft2(orig_sino)
        # corrupt_sino = np.fft.fft2(corrupt_sino)

        self.DataQueryCount += 1
        # print(corrupt_sino.max(), corrupt_sino.min())
        return orig_sino, corrupt_sino, self.checkfeature(corrupt_sino)


if __name__ == "__main__":

    import matplotlib.pylab as plt
    import torch

    def plot_img(img):
        # Pass in the index to read one of the sinogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Original (Sinogram)")
        ax.set_xlabel("Projection position (pixels)")
        ax.set_ylabel("Projection angle (deg)")
        img = np.transpose(img)
        ax.imshow(img)


    import options
    opts = options.parse()

    # traindata = Sino_Dataset('/home/liang/Desktop/Data', 12, testing=False, max_missing=4, min_missing=1)
    # train_loader = torch.utils.data.DataLoader(traindata, batch_size=12, shuffle=True,
    #                                            drop_last=True)

    # test_data = Sino_Dataset(opts.test_file, opts.test_set_size, testing=True,
    #                                   input_depth=opts.channel_number)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    #
    # for orig_sino, corrupt_sino in test_loader:
    #     # Create the input and target tensors from the test set
    #     target_img = orig_sino.clone().detach().cuda()  # torch.tensor(orig_sino, requires_grad=False).cuda()
    #     input_sino = corrupt_sino.clone().detach().cuda()  # torch.tensor(corrupt_sino, requires_grad=False).cuda()
    #
    #     # TODO:Scaling the data 1. Before, 2. After each layer
    #     for idx in range(opts.batch_size):
    #         input_sino[idx] = input_sino[idx] / (input_sino[idx].max() + 1e-8)
    #         target_img[idx] = target_img[idx] / (target_img[idx].max() + 1e-8)
    #
    #     # Run the test batch through the network and accumulate the L1 loss
    #     # During testing the loss for each sinogram is returned every batch
    #     output, loss = network.test(input_sino, target_img)



    # for i in range(2):
    #     for orig_sino, bad_sino in train_loader:
    #         print(orig_sino)
    # good, bad = data[0:100]
    # print(good.shape, bad.shape)
    #
    # print(len(data))
    # for i in range(len(data)):
    #     good, bad = data[i:i+10]
    #     print(good.shape, bad.shape)
    #
    # # for i in range(len(data)):
    # for i in range(len(data)):
    #     good, bad = data[i]
    #     print(good.shape, bad.shape)

    # for i, (good, bad) in enumerate(data):
    #     print(i)
        # print(good.shape)

    plt.show()

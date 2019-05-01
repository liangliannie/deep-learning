#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is used to generate intermediate pickles before the training.


Created on Thu Jun 14 16:19:07 2018
@author: bill
@update: liang
"""

import glob
import numpy as np
import time, pickle
from concurrent.futures import ProcessPoolExecutor, wait
import random


class Sinogram_Processor():

    def __init__(self, data_file, for_test=False, slices=-1, min_blocks=0, max_blocks=0):

        self.for_test = for_test
        self.slices = slices
        self.file = data_file
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        # self.create_masks()


    def process_data(self, file, tof, slices, theta, dist):
        """This function is used to process the sinogram file

            The function will first read the file with the extension .s,
            and find its corresponding good and badblock file, output a stack
            with both good and bad sinograms: Note, here we padding the axis
            to make it more suitable for the network to train, e.g. we make axis
            3 from 50 to 64 by padding wrapped pixels from up and down

            Args:
                file

            Returns:
                output_sino

            Raises:
                IOError: An error occurred accessing the file .
            """
        data = np.fromfile(file, dtype='uint16')
        data = data.reshape((tof, slices, theta, dist))  # Reshape to the file layout

        data = data[:, :, :, :]  # Drop the randoms

        data = np.pad(data, ((0, 0), (0, 0), (1, 0), (0, 0)), 'wrap')
        data = data.reshape((-1, 400, 520))
        # print(data.shape)
        # print(data.shape)
        return data

    def process_sino_file(self):
        """This function is used to process the sinogram file

            The function will first read the file with the extension .s.hdr,
            and find its corresponding good and badblock file, output a stack
            with both good and bad sinograms

            Args:
                self.file

            Returns:
                output_sino_xy

            Raises:
                IOError: An error occurred accessing the file .
            """

        sino_data = []
        corrupt_sino_data = []
        print("working on file {}".format(self.file))
        with open(self.file) as f:
            for line in f:
                if line.startswith("matrix size[1]:="):
                    dist = int(line[16:])
                if line.startswith("matrix size[2]:="):
                    theta = int(line[16:])
                if line.startswith("matrix size[3]:="):
                    slices = int(line[16:])
                if line.startswith('number of scan data types:='):
                    tof = int(line[27:])
                    # print(slices)
        print(tof, slices, theta, dist)
        data_good = self.process_data(self.file[:-4], tof, slices, theta, dist)
        data_bad = self.process_data(self.file[:-8] + '_db-0.s', tof, slices, theta, dist)

        for sino in range(len(data_good)):
            normal_sino = data_good[sino, :, :].astype('uint16') # save as uint16 to save the space of storage in pickle
            corrupt_sino = data_bad[sino, :, :].astype('uint16') # save as uint16 to save the space of storage in pickle
            sino_data.append(normal_sino)
            corrupt_sino_data.append(corrupt_sino)

        sino_data = np.stack(sino_data)

        corrupt_sino_data = np.stack(corrupt_sino_data)
        output_sino_xy = np.stack((sino_data, corrupt_sino_data))

        print("done with file {}".format(self.file))
        # Return a list of two arrays
        return output_sino_xy

    def remove_block(self, orig_img, block_num):
        """This function is omitted."""
        bad_crystal_start = block_num * 13

        bad_img = orig_img

        for p in range(168):
            a_crystal_start = p * 2
            b_crystal_start = (a_crystal_start + 336) % 672

            for l in range(400):
                crystal = (l * 0.5 + a_crystal_start) % 672
                if crystal >= bad_crystal_start and crystal <= (bad_crystal_start + 15):
                    bad_img[p, l] = 0.0

                crystal = (l * 0.5 + b_crystal_start) % 672
                if crystal >= bad_crystal_start and crystal <= (bad_crystal_start + 15):
                    bad_img[p, 399 - l] = 0.0

        return bad_img


def process_sinogram(file):
    """This function is used to process one patient."""
    slices = -1  # Number of slices to pick from sinogram (-1 is all)
    min_blocks = 4
    max_blocks = 4
    for_test = True

    processor = Sinogram_Processor(file, for_test=for_test, slices=slices, min_blocks=min_blocks, max_blocks=max_blocks)

    return processor.process_sino_file()


def main():
    """The main function to call for processing all the data files."""
    print("processing sinogram files")
    count = 0
    for file in sorted(glob.glob("{}*-sino_mash1-0.s.hdr".format('/media/liang/LiangPassport/mash1_sinograms/'))):
        print(file)
        try:
            savename = file[len('/media/liang/LiangPassport/mash1_sinograms/'):-14]
        # for file in sorted(glob.glob("{}*-sino-0.s.hdr".format('/home/liang/Desktop/test/Vision8R_VG75A_NEMA18IQ-Converted/Vision8R_VG75A_NEMA18IQ-LM-00/'))):
        #     if count>=20: #24
        #         continue
            result = process_sinogram(file)
            n = 17
            s = int(result.shape[1]/n)
            for j in range(n):
                with open("/home/liang/Desktop/mash/"+savename+"_{}_{}_dataset.pkl".format(str(count), str(j)), 'wb') as f:
                    pickle.dump(result[:, s*j:s*(j+1), :, :], f, pickle.HIGHEST_PROTOCOL)
            print("File saved:" + "/home/liang/Desktop/mash/"+savename+"_{}_dataset.pkl".format(str(count)))
            count += 1
        except:
            print(file, ' is not saved')
        # time.sleep(10)
    print("All files saved")


if __name__ == "__main__":

    main()





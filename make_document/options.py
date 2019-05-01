#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:31:03 2018

@author: bill
"""
import argparse

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', type=int, dest='visualize', default=True,
                        help='Include this flag to visualize results in visdom (must have it installed)')
    parser.add_argument('--output-path', type=str, dest='output_path', default="/home/liang/Desktop/output",
                        help='Output folder for logs, models, images, etc')
    parser.add_argument('--mask-file', type=str, dest='mask_file', default='/home/liang/Desktop/mask.pkl',
                        help='Path to numpy file containing the mask data')
    parser.add_argument('--training-file', type=str, dest='training_file', default="/home/liang/Desktop/mash1_image/",
                        help='Path to numpy file containing the training data')
    parser.add_argument('--test-file', type=str, dest='test_file', default="/home/liang/Desktop/mash/",
                        help='Path to numpy file containing the test data')
    parser.add_argument('--loading-multiple-files', type=int, dest='loading_multiplefiles', default=True,
                        help='whether loading multiple files for training and testing data')
    parser.add_argument('--load-model', action='store_true', dest='load_model', default=False,
                        help='Include this flag to load a previously trained model located in the output path')
    parser.add_argument('--check-test', type=int, dest='check_test', default=False,
                        help='Include this flag to do test or not')
    parser.add_argument('--epochs', type=int, dest='epochs', default=80000,
                        help='Number of epochs (times through the data) to use in training')
    parser.add_argument('--epoch-size', dest='epoch_size', default=72,
                        help='The number of unique synthetic training images to use for each epoch.')
    parser.add_argument('--test-set-size', dest='test_set_size', default=20,
                        help='The number of samples in the test set')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=5,
                        help='Batch size in the forward pass of the network')
    parser.add_argument('--test-batch-size', type=int, dest='test_batch_size', default=12,
                        help='Test batch size in the forward pass of the network')
    parser.add_argument('--initial-lr', type=float, dest='initial_lr', default=0.0001,
                        help='Initial learning rate parameter (adjustments are adaptive)')
    parser.add_argument('--lr-decay-factor', type=int, dest='lr_decay', default=0.985,
                        help='Factor to reduce the learning rate by at each optimizer step')
    parser.add_argument('--warm-reset-length', type=int, dest='warm_reset_length', default=150,
                        help='The number of training epochs before a warm restart is performed')
    parser.add_argument('--warm-reset-increment', type=int, dest='warm_reset_increment', default=50,
                        help='Increment to the number of epochs before the next warm reset')
    parser.add_argument('--input-channel-number', type=int, dest='input_channel_number', default=1,
                        help='The slicing consistency for sinos, the total number of channels')
    parser.add_argument('--output-channel-number', type=int, dest='output_channel_number', default=1,
                        help='The slicing consistency for sinos, the total number of channels')
    parser.add_argument('--filter-number', type=int, dest='kernel_size', default=5,
                        help='The filer number')
    parser.add_argument('--use-synthetic-data', action='store_true', dest='synthetic_data', default=False,
                        help='Flag to tell the code to generate its own synthetic sinograms. This is the default behavior')
    parser.add_argument('--scan-for-gpu', action='store_true', dest='scan_gpus', default=False,
                        help='When provided this flag tells the software to scan for a free gpu. Useful on multiple gpu systems')
    parser.add_argument('--max-gpus', type=int, dest='max_gpus', default=1,
                        help='Determines the maximum number of GPUs to train on')
    return parser.parse_args()

if __name__ == "__main__":
    parse()
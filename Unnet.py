import torch
from torch.nn import BatchNorm2d as Norm

class UNet_down_block(torch.nn.Module):

    def __init__(self, input_channel, output_channel, down_sample):
        super(UNet_down_block, self).__init__()
        kernel_size = 3
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = Norm(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = Norm(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = Norm(output_channel)
        self.down_sampling = torch.nn.Conv2d(input_channel, input_channel, kernel_size, stride=(2, 2), padding=(1, 1), bias=False)
        self.down_sample = down_sample


    def forward(self, x):
        if self.down_sample:
            x = self.down_sampling(x)
        x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        return x

class UNet_up_block(torch.nn.Module):

    def __init__(self, prev_channel, input_channel, output_channel, ID):
        super(UNet_up_block, self).__init__()
        kernel_size = 3
        self.ID = ID
        self.up_sampling = torch.nn.ConvTranspose2d(input_channel, input_channel, 4, stride=(2, 2), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn1 = Norm(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn2 = Norm(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn3 = Norm(output_channel)


    def forward(self, prev_feature_map, x):

        if self.ID == 1:
            x = self.up_sampling(x)
        elif self.ID == 2:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        elif self.ID == 3:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='area') #‘nearest’ | ‘linear’ | ‘bilinear’ | ‘trilinear’ | ‘area’
        x = torch.cat((x, prev_feature_map), dim=1)
        # x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        # x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        # x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv1(x)), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv3(x)), 0.2)
        return x


class UNet_up_block2(torch.nn.Module):

    def __init__(self, prev1_channel, prev2_channel, input_channel, output_channel, ID):
        super(UNet_up_block2, self).__init__()
        kernel_size = 3
        self.ID = ID
        self.up_sampling = torch.nn.ConvTranspose2d(input_channel, input_channel, 4, stride=(2, 2), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(prev1_channel+prev2_channel+ input_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn1 = Norm(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn2 = Norm(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=(1, 1), padding=(1, 1), bias= False)
        self.bn3 = Norm(output_channel)


    def forward(self, prev_feature_map1,prev_feature_map2, x):

        if self.ID == 1:
            x = self.up_sampling(x)
        elif self.ID == 2:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='nearest')
        elif self.ID == 3:
            x = torch.nn.functional.interpolate(x, scale_factor=(2, 2), mode='area') #‘nearest’ | ‘linear’ | ‘bilinear’ | ‘trilinear’ | ‘area’
        x = torch.cat((x, prev_feature_map1, prev_feature_map2), dim=1)
        # x = torch.nn.functional.leaky_relu(self.bn1((self.conv1(x))), 0.2)
        # x = torch.nn.functional.leaky_relu(self.bn2((self.conv2(x))), 0.2)
        # x = torch.nn.functional.leaky_relu(self.bn3((self.conv3(x))), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv1(x)), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu((self.conv3(x)), 0.2)
        return x


class UNet(torch.nn.Module):

    def __init__(self, opts):
        super(UNet, self).__init__()

        self.opts = opts
        input_channel_number = opts.input_channel_number
        output_channel_number = opts.output_channel_number
        # kernel_size = opts.kernel_size # we could change this later
        kernel_size = 3

        # self.linear1 = torch.nn.Sequential(*self.lin_tan_drop(400 * 520, 400 * 520, 64))
        # self.linear2 = torch.nn.Sequential(*self.lin_tan_drop(200 * 260, 200 * 260, 64))
        #
        self.linear1 = torch.nn.Sequential(*self.lin_tan_drop(440 * 440, 440 * 440, 64))
        self.linear2 = torch.nn.Sequential(*self.lin_tan_drop(220 * 220, 220 * 220, 64))
        # self.linear1 = torch.nn.Sequential(*self.lin_tan_drop(400 * 520, 400 * 520, 64))
        # self.linear2 = torch.nn.Sequential(*self.lin_tan_drop(200 * 260, 200 * 260, 64))
        # self.linear3 = torch.nn.Sequential(*self.lin_tan_drop(100 * 130, 100 * 130, 64))

        # Encoder network
        self.down_block1 = UNet_down_block(input_channel_number, 64, False) # 64*520
        self.down_block2 = UNet_down_block(64, 128, True) # 64*520
        self.down_block3 = UNet_down_block(128, 256, True) # 64*260


        # bottom convolution
        self.mid_conv1 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn1 = Norm(256)
        self.mid_conv2 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn2 = Norm(256)
        self.mid_conv3 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False) #, dilation=4 # 64*260
        self.bn3 = Norm(256)
        self.mid_conv4 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn4 = Norm(256)
        self.mid_conv5 = torch.nn.Conv2d(256, 256, kernel_size, padding=(1, 1), bias=False)# 64*260
        self.bn5 = Norm(256)

        # Decoder network
        # self.up_block2 = UNet_up_block(128, 256, 128, 1)# 64*520
        # self.up_block3 = UNet_up_block(64, 128, 64, 1)# 64*520
        #
        self.up_block21 = UNet_up_block2(128, 128, 256, 128, 1)# 64*520
        self.up_block31 = UNet_up_block2(64, 64, 128, 64, 1)# 64*520
        # Final output
        self.last_conv1 = torch.nn.Conv2d(64, 64, 3, padding=(1, 1), bias=False)# 64*520
        self.last_bn = Norm(64) #
        self.last_conv2 = torch.nn.Conv2d(64, output_channel_number, 3, padding=(1, 1))# 64*520
        self.last_bn2 = Norm(output_channel_number) # 64*520

        self.softplus = torch.nn.Softplus(beta=5, threshold=100)
        self.relu = torch.nn.ReLU()
        self.tanhshrink = torch.nn.Tanhshrink()
        self.tanh = torch.nn.Tanh()

    def lin_tan_drop(self, num_features_in, num_features_out, kernel_features, dropout=0.5):
        layers = []
        layers.append(torch.nn.Linear(num_features_in, kernel_features, bias=True))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(kernel_features, num_features_out, bias=True))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Dropout(p=dropout))
        return layers

    def forward(self, x, test=False):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)

        x4 = torch.nn.functional.leaky_relu(self.bn1(self.mid_conv1(x3)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn2(self.mid_conv2(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn3(self.mid_conv3(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.mid_conv4(x4)), 0.2)
        x4 = torch.nn.functional.leaky_relu(self.bn5(self.mid_conv5(x4)), 0.2)
        s1, s2, s3 = x1.shape, x2.shape, x4.shape

        x1_l = self.linear1(x1.reshape(s1[0]*s1[1], -1)).reshape(x1.shape)
        x2_l = self.linear2(x2.reshape(s2[0]*s2[1], -1)).reshape(x2.shape)
        # x4_l = self.linear3(x4.reshape(s3[0]*s3[1], -1)).reshape(x4.shape)

        # out = self.up_block2(x2, x4)
        # out = self.up_block3(x1, out)
        out = self.up_block21(x2_l, x2, x4)
        out = self.up_block31(x1_l, x1, out)

        out = torch.nn.functional.relu(self.last_conv1(out))
        # out = torch.nn.functional.relu(self.last_bn2(self.last_conv2(out)))
        out = self.last_conv2(out)
        out = self.relu(out)
        return out


if __name__ == '__main__':

    net = UNet()
    x = torch.randn(1, 159, 220, 220)
    out = net(x)
    print(out.shape)

    # framelets = Framelets()
    # framelets.cuda()
    # x = torch.autograd.Variable(torch.rand(1, 1, 64, 520).cuda())
    #
    # # x = torch.autograd.Variable(torch.rand(1, 3, 256, 256))
    # out = framelets(x)
    #
    # print(out.shape)
    # print(out)


    # Load image
    import numpy as np
    import matplotlib.pylab as plt
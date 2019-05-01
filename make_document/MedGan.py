import torch
import torchvision.models as models


class FeatureExtractor(torch.nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        l_style = []
        l_content = []
        for name, module in self.submodule._modules.items():

            # with torch.no_grad():
            x = module(x.clone())

            if self.extracted_layers in module.__str__():
                gram_matrix = torch.mean(x*x, (2,3))
                gram_matrix = torch.mm(gram_matrix.t(), gram_matrix)
                l_content.append(x)
                l_style.append(gram_matrix)

        return l_style, l_content


def medganloss(input, target, model='VGG', lamb=1):

    if model == 'VGG':
        pretrained_network = models.vgg16(pretrained=True)
    elif model == 'ResNet':
        print('TODO add')
        pretrained_network = models.resnet18(pretrained=True)
    elif model == 'AlexNet':
        print('TODO add')
        pretrained_network = models.alexnet(pretrained=True)
    elif model == 'DenseNet':
        print('TODO add')
        pretrained_network = models.densenet161(pretrained=True)
    else:
        print('TODO add')
        pretrained_network = models.googlenet(pretrained=True)


    if input.is_cuda:
        pretrained_network.cuda()
    features = FeatureExtractor(pretrained_network.features, 'Conv2d')

    l_s_x, l_c_x = features(input)
    l_s_y, l_c_y = features(target)

    lambda_s = torch.randn(len(l_s_x), 1)
    lambda_c = torch.randn(len(l_s_y), 1)

    if input.is_cuda:
        lambda_s = lambda_s.cuda()
        lambda_c = lambda_c.cuda()

    l_s = torch.sum(torch.cat([(torch.abs(c)*torch.norm((a-b), 2)/(4*torch.numel(a)**2)) for a, b, c in zip(l_s_x, l_s_y, lambda_s)]))
    l_c = torch.sum(torch.cat([(torch.abs(c)*torch.norm((a-b), 2)/(torch.numel(a))) for a, b, c in zip(l_s_x, l_s_y, lambda_c)]))


    loss = (l_s + lamb*l_c)

    if input.is_cuda:
        loss = loss.cuda()

    del l_s_y, l_s_x, l_c_x, l_c_y, l_s, l_c

    return loss




class MedGanloss(torch.nn.Module):

    def __init__(self, window_size=11):
        super(MedGanloss, self).__init__()
        self.window_size = window_size

    def forward(self, input, target):
        shape = input.shape
        if shape[1] == 1:
            input = input.expand(shape[0], 3, shape[2], shape[3])
            target = target.expand(shape[0], 3, shape[2], shape[3])
        return medganloss(input, target, model='VGG')



if __name__ == '__main__':

    x = torch.randn(1, 1, 64, 520).cuda()
    y = torch.randn(1, 1, 64, 520).cuda()
    l1 = MedGanloss()

    loss = l1(x,y)

    print(loss)
    x = torch.randn(1, 1, 64, 520)
    y = torch.randn(1, 1, 64, 520)
    l1 = MedGanloss()

    loss = l1(x,y)

    print(loss)
    # model=Discriminator()
    # z= model(x)
    # print(z.shape)

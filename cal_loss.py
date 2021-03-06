import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
'''
Mainly copied from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
Slightly modified to calculate perceptual loss.(Modified the implementation of VGG class)
'''

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def single_gram(image):
    a, b, c, d = image.size()
    features = image.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d * 2)

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        '''
        Modified for vgg_19 to cal perceptual loss
        Perceptual loss:
        [conv1 2’, ‘conv2 2’, ‘conv3 2’, ‘conv4 2’, and ‘conv5 2’]
        :param features:
        :param num_classes:
        :param init_weights:
        '''
        super(VGG, self).__init__()

        self.mean = torch.tensor([[0.485, 0.456, 0.406]], dtype = torch.float)
        self.std = torch.tensor([[0.229, 0.224, 0.225]], dtype = torch.float)
        self.perceptual_dict = {
            'inp': 256 * 256 * 3,
            'features.2': 256*512*64,
            'features.7': 128*256*128,
            'features.12': 64*128*256,
            'features.21': 32*64*512,
            'features.30': 16*32*512
        }

        self.features = features
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        '''
        if init_weights:
            self._initialize_weights()

    def forward(self, output, img):

        '''
        # modifiled for vgg19 to cal perceptual loss;

        :param output:   output of the generative model
        :img: raw img to calculate perceptual loss
        :return:
        '''

        mean = self.mean.cuda()
        std = self.std.cuda()


        f_out = (output.permute(0, 2, 3, 1) - mean) / std
        f_out = f_out.permute(0, 3, 1, 2)

        f_img = (img.permute(0, 2, 3, 1) - mean) / std
        f_img = f_img.permute(0, 3, 1, 2)

        f_img.volatile = True
        f_out.volatile = True

        perceptual = torch.zeros([1, 5], dtype = torch.float).cuda()
        #style = torch.zeros([1, 5], dtype = torch.float).cuda()

        i = 0
        for name, m in self.named_modules():
            if name in ['', 'features', 'classifier']:
                continue
            if 'classifier' in name:
                continue
            #print(name)
            f_out = m(f_out)
            f_img = m(f_img)

            if name in self.perceptual_dict.keys():

                pe = torch.abs((f_img - f_out)).mean(dim = 1)
                #spe = torch.sum(((single_gram(f_img) - single_gram(f_out))) ** 2)
                perceptual[0][i] = torch.mean(torch.mean(torch.mean(pe), dim = -1), dim = -1)
                #style[0][i] = spe
                i += 1


                f_out.detach_()
                f_img.detach_()
                del pe
                #del spe
                if i >= 5:
                    break

        #perceptual.detach_()
        return perceptual#, style#torch.tensor(perceptual, dtype = torch.float)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict = False)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def test():

    vgg = vgg19(True)

    vgg.cuda()
    vgg.eval()

    img = torch.randn((1, 3, 224, 224)).type(torch.float).cuda()
    out = torch.randn((1, 3, 224, 224)).type(torch.float).cuda()

    loss = vgg(out, img)

    print(loss)



class tv_loss(nn.Module):
    '''
    Calculating tv loss in hole region
    L1 tv with mean in all dims

    mask should be 0/1 where 1 was mask!!
    '''

    def __init__(self):
        super(tv_loss, self).__init__()

    def forward(self, outputs, labels, masks):
        outputs_hole = outputs * masks
        targets_hole = labels * masks
        tv_loss = torch.sum(torch.abs(outputs_hole[:, :, :, 1:] - targets_hole[:, :, :, :-1])) \
                  + torch.sum(torch.abs(outputs_hole[:, :, 1:, :] - targets_hole[:, :, :-1, :]))
        tv_loss /= outputs.size(0)

        return tv_loss

class mask_l1(nn.Module):
    '''
    output: scalar;   mean in all dims?
    '''

    def __init__(self):
        super(mask_l1, self).__init__()

    def forward(self, outputs, labels, masks):
        '''

        :param outputs:
        :param labels:
        :param masks: 0/1  where 1 was masked
        :return:
        '''
        hole_loss = F.l1_loss(outputs * masks, labels * masks)

        return hole_loss

class other_l1(nn.Module):
    '''
    output: scalar;   mean in all dims?
    '''
    def __init__(self):
        super(other_l1, self).__init__()

    def forward(self, outputs, labels, masks):
        '''
        :param outputs:
        :param labels:
        :param masks: 0/1  where 1 was masked
        :return:
        '''
        other_loss = F.l1_loss(outputs * (1 - masks), labels * (1 - masks))

        return other_loss
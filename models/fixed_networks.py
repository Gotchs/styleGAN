import torch.nn as nn
import utils.module_utils as mutils

#############
# functions
#############

def dc_Gen_R_BN():
    '''
    Fixed implementation of DCGAN generator
    Relu before Batchnorm
    Structure reference: https://arxiv.org/abs/1511.06434
    '''
    return nn.Sequential(
        nn.Linear(100, 1024 * 4 * 4),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024 * 4 * 4),
        mutils.Unflatten(-1, 1024, 4, 4),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.Tanh()
    )

def dc_Dis_LR_BN():
    '''
    Fixed implementation of DCGAN discriminator
    LeakyRelu before Batchnorm
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(512),
        mutils.Flatten(),
        nn.Linear(512 * 4 * 4, 1),
        nn.Sigmoid()
    )

def ls_Gen_IN_R():
    pass

def ls_Dis_IN_LR():
    pass
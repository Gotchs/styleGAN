from .baseGAN import BaseGAN
from . import fixed_networks as fnet

##############
# classes
##############

class fixed_DCGAN(BaseGAN):

    def __init__(self, useGPU=True):
        super(fixed_DCGAN, self).__init__(useGPU, 64, 100)


    def get_networks(self, net_type='BN_R'):
        '''
        Set generator and discriminator, default DCGAN with batchnorm
        '''
        if net_type == 'BN_R':
            self.G = fnet.dc_Gen_BN_R().type(self.type)
            self.D = fnet.dc_Dis_BN_LR().type(self.type)
        elif net_type == 'R_BN':
            self.G = fnet.dc_Gen_R_BN().type(self.type)
            self.D = fnet.dc_Dis_LR_BN().type(self.type)
        elif net_type == 'IN_R':
            self.G = fnet.dc_Gen_IN_R().type(self.type)
            self.D = fnet.dc_Dis_IN_LR().type(self.type)
        else:
            raise NotImplementedError('Net type [%s] is not implemented' % net_type)
        self.net_type = net_type


import argparse
from models.fixed_GANs import LSGAN_Res

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--default', action='store_true', help='Train from zero using default training strategy.')
parser.add_argument('-p', '--ckp_route', type=str, default=None, help='Route of checkpoint file.')
parser.add_argument('-c', '--cpu', action='store_false', help='Using CPU for training. (Not recommend)')
parser.add_argument('--net_type', type=str, default='BN_R', help='Net type to use. Default: BN_R')
parser.add_argument('--ResGroup_size', type=int, default=2, help='# of ResBlocks to use. Default: 2')
parser.add_argument('--init_type', type=str, default='kaiming', help='Method for weight initialization. Default: kaiming')
parser.add_argument('--gain', type=float, default=0.02, help='An scaling factor for weight initialization. Default: 0.02')
parser.add_argument('--dataset', type=str, default='LSUN', help='Name of dataset. Default: LSUN')
parser.add_argument('--classes', type=str, default='church_outdoor_train', help='Which classes of LSUN to use. Default: church_outdoor_train')
parser.add_argument('--batch_size', type=int, default=128, help='Size of minibatch. Default: 128')
parser.add_argument('--optim_G', type=str, default='Adam', help='Optimizer for generator. Default: Adam')
parser.add_argument('--lr_G', type=float, default=2e-4, help='Learning rate of generator. Default: 2e-4')
parser.add_argument('--beta1_G', type=float, default=0.5, help='Beta1 for Adam optimizer of generator. Default: 0.5')
parser.add_argument('--beta2_G', type=float, default=0.999, help='Beta2 for Adam optimizer of generator. Default: 0.999')
parser.add_argument('--optim_D', type=str, default='Adam', help='Optimizer for discriminator. Default: Adam')
parser.add_argument('--lr_D', type=float, default=2e-4, help='Learning rate of discriminator. Default: 2e-4')
parser.add_argument('--beta1_D', type=float, default=0.5, help='Beta1 for Adam optimizer of discriminator. Default: 0.5')
parser.add_argument('--beta2_D', type=float, default=0.999, help='Beta2 for Adam optimizer of discriminator. Default: 0.999')
parser.add_argument('--loss', type=str, default='LSGAN', help='Loss function. Default: LSGAN')
parser.add_argument('--hard_label', action='store_false', help='Using hard label.')
parser.add_argument('--threshold_D', type=float, default=0.2, help='Threshold for training D. Default: 0.2')
parser.add_argument('--epoch', type=int, default=8, help='Number of epochs to train. Default: 8')
parser.add_argument('--show_every', type=int, default=250, help='Display (save or show) images and loss periodically. Default: 250')
parser.add_argument('--lr_decay_every', type=int, default=1000, help='Decay learning rate periodically. Default: 1000')
parser.add_argument('--G_decay', type=float, default=0.9, help='lr_decay coefficient for generator. Default: 0.9')
parser.add_argument('--D_decay', type=float, default=0.85, help='lr_decay coefficient for discriminator. Default: 0.85')
parser.add_argument('--model_route', type=str, default='./savemodels/', help='Route for saving models. Default: ./savemodels/')
parser.add_argument('--figure_route', type=str, default='./savefigs/', help='Route for saving figures. Default: ./savefigs/')
opt = parser.parse_args()

if __name__ == '__main__':
    if opt.default:
        net = LSGAN_Res()
        net.get_networks()
        net.init_weights()
        net.train()
    elif opt.ckp_route is not None:
        net = LSGAN_Res(opt.cpu)
        net.train(ckp_route=opt.ckp_route, 
                  num_epochs=opt.epoch, show_every=opt.show_every, threshold_D=opt.threshold_D,
                  lr_decay_every=opt.lr_decay_every, G_decay=opt.G_decay, D_decay=opt.D_decay,
                  model_route=opt.model_route, figure_route=opt.figure_route)
    else:
        net = LSGAN_Res(opt.cpu)
        net.get_networks(net_type=opt.net_type, ResGroup_size=opt.ResGroup_size)
        net.init_weights(init_type=opt.init_type, gain=opt.gain)
        net.get_dataset(dset_name=opt.dataset, classes=[opt.classes])
        net.get_dataloader(batch_size=opt.batch_size)
        net.get_G_optimizer(optim_name=opt.optim_G, lr=opt.lr_G, betas=(opt.beta1_G, opt.beta2_G))
        net.get_D_optimizer(optim_name=opt.optim_D, lr=opt.lr_D, betas=(opt.beta1_D, opt.beta2_D))
        net.get_loss(loss_name=opt.loss, soft_label=opt.hard_label)
        net.train(num_epochs=opt.epoch, show_every=opt.show_every, threshold_D=opt.threshold_D,
                  lr_decay_every=opt.lr_decay_every, G_decay=opt.G_decay, D_decay=opt.D_decay,
                  model_route=opt.model_route, figure_route=opt.figure_route)

# -*- coding: utf-8 -*-


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # dataset parameters specified in the training procedure
        parser.add_argument('--shuffle', type=int, default=1, help='whether use the shuffle')

        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', type=bool, default=False, help='whether saves model by iteration')
        parser.add_argument('--continue_train', type=int, default=0, help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--optim_mode', type=str, default='Adam', help='Adam,SGD,etc')
        parser.add_argument('--init_lr', default=0.002, type=float, help='the initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='multistep',
                            help='learning rate policy. [linear | step | plateau | cosine | multistep]')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay_iters', type=str, default='400',
                            help='multiply by a gamma every lr_decay_iters iterations, e.g., 400 | 1000')
        parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum of SGD')
        parser.add_argument('--sgd_weight_decay', type=float, default=0.0, help='weight decay of SGD')
        parser.add_argument('--adam_beta1', type=float, default=0.5, help='beta1 of Adam')
        parser.add_argument('--loss_mode', type=str, default='DwF', help='BCE,DICE,CE,CBL,GDL,etc')
        parser.add_argument('--total_epoch', type=int, default=600, help='how many epochs to train, e.g., 600 | 1200')
        parser.add_argument('--epoch_interval', type=int, default=20, help='epoch interval')

        self.isTrain = True
        return parser

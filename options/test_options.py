# -*- coding: utf-8 -*-

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # dataset parameters specified in the test procedure
        parser.set_defaults(shuffle=0)
        parser.set_defaults(batch_size=1)

        # test parameters
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--threshold', type=float, default= 0.5, help='threshold for the sigmoid during the test procedure')
        parser.add_argument('--eval',type=bool,default=True, help='use eval mode during test time.')
        self.isTrain = False
        return parser

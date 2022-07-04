# -*- coding: utf-8 -*-

import os
import time
import sys

import random
import numpy as np
import torch

from main_code.options.train_options import TrainOptions
from main_code.dataloader import create_dataset
from main_code.models import create_model
from main_code.util.visualizer import Visualizer

EPOCH_THRESH = 5


if __name__ == "__main__":
    # ==============================option/dataset/visualizer===============================
    opt = TrainOptions().parse()
    train_dataset = create_dataset(opt)
    print('The number of training images = %d' % len(train_dataset))
    visualizer = Visualizer(opt)
    # ==============================option/dataset/visualizer===============================

    # =====================================model setting====================================
    model = create_model(opt)
    model.setup(opt)
    model.train()
    # =====================================model setting====================================

    # ========================================variable=======================================
    total_iters = 0
    # ========================================variable=======================================

    # ====================================Train Procedure====================================
    for epoch in range(opt.epoch_count, opt.total_epoch + 1):
        CONDITIONFlAG_1 = (epoch % opt.epoch_interval == 0)
        _epoch = int(epoch / opt.epoch_interval)

        if (CONDITIONFlAG_1):
            epoch_start_time = time.time()
            batch_train_loss = []

        for idx, data in enumerate(train_dataset):
            total_iters += opt.batch_size
            model.set_input(tuple(data))
            model._set_use_fla_status(_epoch, epoch_thresh=EPOCH_THRESH)
            model.optimize_parameters()
            step_loss = model.get_current_losses()
            batch_loss = visualizer.save_step_losses(total_iters, step_loss)
            visualizer.print_step_losses(total_iters, step_loss)
            batch_train_loss.append(batch_loss)

        if (CONDITIONFlAG_1):
            batch_train_loss_array = np.asarray(batch_train_loss, dtype=np.float)
            epoch_loss = np.sum(batch_train_loss_array, axis=0) / batch_train_loss_array.shape[0]
            epoch_loss = model.get_current_epoch_losses(epoch_loss)
            visualizer.print_epoch_losses(_epoch, epoch_loss)
            model.print_lr()
            print('End of epoch %d / %d \t Time Taken: %d sec' % (
                _epoch, int(opt.total_epoch / opt.epoch_interval),
                (time.time() - epoch_start_time) * opt.epoch_interval))
            model.save_networks(_epoch)
            print('Save the %d th model!' % (_epoch))
        model.update_learning_rate()
    # ====================================Train Procedure====================================




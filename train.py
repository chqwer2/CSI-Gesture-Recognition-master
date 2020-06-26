# coding=utf-8
from __future__ import absolute_import, print_function    #python的向上兼容
import time
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
import numpy as np
import os.path as osp

import models
import losses
import DataSet

from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display  #将一些常用操作/函数封装在utils中
from utils.serialization import save_checkpoint, load_checkpoint
from trainer import train
from utils import orth_reg


cudnn.benchmark = True

use_gpu = True
losses_ = []

def main(args):
    # s_ = time.time()

    save_dir = args.save_dir          #模型存储位置
    mkdir_if_missing(save_dir)        #检查该存储文件是否可用/utils库

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)                                                   #打印当前训练模型的参数
    start = 0

    model = models.create(args.net, pretrained = False , model_path = None, normalized = True)   #@@@创建模型/ pretrained = true 将会去读取现有预训练模型/models文件中的函数


    model = torch.nn.DataParallel(model)    #使用torch进行模型的并行训练/分布
    model = model.cuda()                    #使用GPU

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)              #优化器

    criterion = losses.create(args.loss, margin_same=args.margin_same, margin_diff=args.margin_diff).cuda()  #TWConstrativeloss

    data = DataSet.create(name = args.data, root=args.data_root, set_name = args.set_name)  #数据 set_name = "test" or "train" ;

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,shuffle = True,
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    for epoch in range(start, 50): #args.epochs

        L = train(epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args)
        losses_.append(L)


        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

    # added
    batch_nums = range(1, len(losses_) + 1)
    import matplotlib.pyplot as plt
    plt.plot(batch_nums, losses_)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=256, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 32')

    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')

    parser.add_argument('--margin_same', default=0.7, type=float,
                        help='margin_same in loss function')
    parser.add_argument('--margin_diff', default=0.4, type=float,
                        help='margin_diff in loss function')

    parser.add_argument('--data', default='sign', required=False,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default="../IoT/DataSet/",
                        help='path to Data Set')

    #you can usinng 'Simple-Net', 'ResNet', 'VGG'
    parser.add_argument('--net', default='VGG')

    parser.add_argument('--loss', default='TMContrastiveLoss', required=False,
                        help='loss for training network')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=30, type=int, metavar='N',
                        help='number of epochs to save model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')


    parser.add_argument('--set_name', default="train",
                        help='training set or testing set')
    parser.add_argument('--save_dir', default="./models",
                        help='where the trained models save')

    parser.add_argument('--nThreads', '-j', default=1, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    main(parser.parse_args()) #与run_train_00.sh使用




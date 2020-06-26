# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from utils import AverageMeter
from evaluations import Recall_at_ks, pairwise_similarity
from utils.serialization import load_checkpoint
import torch
import ast
import time



def main(args):
    batch_time = AverageMeter()
    end = time.time()

    checkpoint = load_checkpoint(args.resume)               #loaded
    print('pool_features:',args.pool_feature)
    epoch = checkpoint['epoch']

    gallery_feature, gallery_labels, query_feature, query_labels = \
    Model2Feature(data=args.data, root=args.data_root, net=args.net, checkpoint=checkpoint
    , batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature)    #output


    sim_mat = pairwise_similarity(query_feature, gallery_feature)    #成对相似性
    if args.gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    print('labels',query_labels)
    print('feature:',gallery_feature)


    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

    result = '  '.join(['%.4f' % k for k in recall_ks])   #   result=recall_ks
    print('Epoch-%d' % epoch, result)
    batch_time.update(time.time() - end)

    print('Epoch-%d\t' % epoch,
          'Time {batch_time.avg:.3f}\t'.format
          ( batch_time=batch_time ))


    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    similarity = torch.mm(gallery_feature, gallery_feature.t())
    similarity.size()

    #draw Feature Map
    img = torchvision.utils.make_grid(similarity).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Testing')

    parser.add_argument('--data', type=str, default='sign')
    parser.add_argument('--data_root', type=str, default="../IoT/DataSet")
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=True,help='Is gallery identical with query')

    parser.add_argument('--net', type=str, default='Simple-Net')
    parser.add_argument('--resume', '-r', type=str, default="../IoT/models/ckp_ep30.pth.tar", metavar='PATH')  #checkpoint的绝对路径

    #parser.add_argument('--dim', '-d', type=int, default=512,help='Dimension of Embedding Feather')
    #parser.add_argument('--width', type=int, default=224,help='width of input image')
    parser.add_argument('--set_name', default="test",  help='training set or testing set')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nThreads', '-j', default=0, type=int, metavar='N',help='number of data loading threads (default: 2)')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,help='if True extract feature from the last pool layer')
    main(parser.parse_args())
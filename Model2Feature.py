# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint,  root=None, nThreads=16, batch_size=100, pool_feature=False, **kargs):
    dataset_name = data
    model = models.create(net, pretrained = False ,  normalized = True)
    # resume = load_checkpoint(ckp_path)
    resume = checkpoint

    model.load_state_dict(resume['state_dict'])
    model.eval()
    model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(name = data, root=root, set_name = 'test')
    
    if dataset_name in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=nThreads)

        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)


    else:#here

        print('using else')
        data_loader = torch.utils.data.DataLoader(
            data.test, batch_size=batch_size, shuffle=True,
            drop_last=True, pin_memory=True, num_workers=nThreads)

        features, labels = extract_features(model, data_loader,  pool_feature=pool_feature)

        #全等？
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    return gallery_feature, gallery_labels, query_feature, query_labels


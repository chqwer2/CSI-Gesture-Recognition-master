from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable


#magin_same is the least similarity value of same class samples margin_diff is the upper bound similarity value of different classes samples
class TwoMarginContrastiveLoss(nn.Module):
    def __init__(self, margin_same,margin_diff,**kwargs):
        super(TwoMarginContrastiveLoss, self).__init__()
        self.margin_same = margin_same #同类的相似性的最小值 (0,1)
        self.margin_diff = margin_diff #不同类的相似性最大值

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            #find the positive pairs(from the same class)
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])
            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)

            #find the negative pairs(from different classes)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin_diff)
            pos_pair = torch.masked_select(pos_pair_,pos_pair_ < self.margin_same)
            
            neg_loss = 0

            pos_loss = 0
            if len(pos_pair) > 0:
                pos_loss = torch.sum(1-pos_pair)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)

            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim



def main():
    data_size = 32
    input_dim = 3
    output_dim = 20
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(TwoMarginContrastiveLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')



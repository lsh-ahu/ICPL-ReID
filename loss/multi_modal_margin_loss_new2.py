import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, feat3, label1):
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        feat3 = feat3.chunk(label_num, 0)
        total_dist = 0.0
        for i in range(label_num):
          center1 = torch.mean(feat1[i], dim=0)
          center2 = torch.mean(feat2[i], dim=0)
          center3 = torch.mean(feat3[i], dim=0)
          dist = max(abs(self.margin[0] - self.dist(center1, center2)), 
                  abs(self.margin[1] - self.dist(center2, center3)), 
                  abs(self.margin[2] - self.dist(center1, center3)))

          if self.dist_type == 'l2' or self.dist_type == 'l1':
            if i == 0:
              total_dist = dist
            else:
              total_dist += dist
        return total_dist



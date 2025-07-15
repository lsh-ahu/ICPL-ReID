import numpy as np
from torch import nn
import torch
from torch.autograd import Variable

class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()

		self.power = 2
	
	def forward(self, feat1, feat2, feat3, label1, label2):

		norm = feat1.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
		feat1 = feat1.div(norm)

		norm = feat2.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
		feat2 = feat2.div(norm)

		norm = feat3.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
		feat3 = feat3.div(norm)

		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		feat3 = feat3.chunk(label_num, 0)

		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			center3 = torch.mean(feat3[i], dim=0)

			center_list = [center1, center2, center3]
			dist_tmp = 0.
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist_tmp += max(0, self.dist(center_list[0], center_list[1]) - self.margin)
					dist_tmp += max(0, self.dist(center_list[0], center_list[2]) - self.margin)
					dist_tmp += max(0, self.dist(center_list[1], center_list[2]) - self.margin)
					# dist = dist_tmp / 3.
					dist = dist_tmp / 3.
				else:
					dist_tmp += max(0, self.dist(center_list[0], center_list[1]) - self.margin)
					dist_tmp += max(0, self.dist(center_list[0], center_list[2]) - self.margin)
					dist_tmp += max(0, self.dist(center_list[1], center_list[2]) - self.margin)
					dist += dist_tmp / 3.
			elif self.dist_type == 'cos':
				if i == 0:
					dist_tmp += max(0, 1-self.dist(center_list[0], center_list[1]) - self.margin)
					dist_tmp += max(0, 1-self.dist(center_list[0], center_list[2]) - self.margin)
					dist_tmp += max(0, 1-self.dist(center_list[1], center_list[2]) - self.margin)
					dist = dist_tmp / 3.
				else:
					dist_tmp += max(0, 1-self.dist(center_list[0], center_list[1]) - self.margin)
					dist_tmp += max(0, 1-self.dist(center_list[0], center_list[2]) - self.margin)
					dist_tmp += max(0, 1-self.dist(center_list[1], center_list[2]) - self.margin)
					dist += dist_tmp / 3.
		return dist
		
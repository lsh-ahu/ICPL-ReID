from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from scipy.optimize import linear_sum_assignment


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_nir, inputs_tir, targets, features_rgb, features_nir, features_tir, momentum):
        ctx.features_rgb = features_rgb
        ctx.features_nir = features_nir
        ctx.features_tir = features_tir
        ctx.momentum = momentum

        ctx.save_for_backward(inputs_rgb, inputs_nir, inputs_tir, targets)
        outputs_rgb = inputs_rgb.mm(ctx.features_rgb.t())
        outputs_nir = inputs_nir.mm(ctx.features_nir.t())
        outputs_tir = inputs_tir.mm(ctx.features_tir.t())

        return outputs_rgb, outputs_nir, outputs_tir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2, grad_outputs3):
        inputs_rgb, inputs_nir, inputs_tir, targets = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None
        grad_inputs2 = None

        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features_rgb)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features_nir)
        if ctx.needs_input_grad[2]:
            grad_inputs3 = grad_outputs3.mm(ctx.features_tir)

        # momentum update
        for x,y in zip(inputs_rgb, targets):
            ctx.features_rgb[y] = ctx.momentum * ctx.features_rgb[y] + (1. - ctx.momentum) * x
            ctx.features_rgb[y] /= ctx.features_rgb[y].norm()

        for x, y in zip(inputs_nir, targets):
            ctx.features_nir[y] = ctx.momentum * ctx.features_nir[y] + (1. - ctx.momentum) * x
            ctx.features_nir[y] /= ctx.features_nir[y].norm()

        for x, y in zip(inputs_tir, targets):
            ctx.features_tir[y] = ctx.momentum * ctx.features_tir[y] + (1. - ctx.momentum) * x
            ctx.features_tir[y] /= ctx.features_tir[y].norm()

        return grad_inputs1, grad_inputs2, grad_inputs3, None, None, None, None, None



def cm(inputs_rgb, inputs_nir, inputs_tir, indexes, features_rgb, features_nir, features_tir, momentum):
    return CM.apply(inputs_rgb, inputs_nir, inputs_tir, indexes, features_rgb, features_nir, features_tir, torch.Tensor([momentum]).to(inputs_rgb.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features_rgb', torch.zeros(num_samples, num_features))
        self.register_buffer('features_nir', torch.zeros(num_samples, num_features))
        self.register_buffer('features_tir', torch.zeros(num_samples, num_features))


    def forward(self, inputs_rgb, inputs_nir, inputs_tir, targets):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_nir = F.normalize(inputs_nir, dim=1).cuda()
        inputs_tir = F.normalize(inputs_tir, dim=1).cuda()

        outputs_rgb, outputs_nir, outputs_tir = cm(inputs_rgb, inputs_nir, inputs_tir, targets, self.features_rgb, self.features_nir, self.features_tir, self.momentum)
        outputs_rgb /= self.temp
        outputs_nir /= self.temp
        outputs_tir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets)
        loss_nir = F.cross_entropy(outputs_nir, targets)
        loss_tir = F.cross_entropy(outputs_tir, targets)

        return loss_rgb, loss_nir, loss_tir



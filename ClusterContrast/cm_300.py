from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from scipy.optimize import linear_sum_assignment


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_nir, targets, features_rgb, features_nir, momentum):
        ctx.features_rgb = features_rgb
        ctx.features_nir = features_nir
        ctx.momentum = momentum

        ctx.save_for_backward(inputs_rgb, inputs_nir, targets)
        outputs_rgb = inputs_rgb.mm(ctx.features_rgb.t())
        outputs_nir = inputs_nir.mm(ctx.features_nir.t())

        return outputs_rgb, outputs_nir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs_rgb, inputs_nir, targets = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None

        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features_rgb)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features_nir)

        # momentum update
        for x,y in zip(inputs_rgb, targets):
            ctx.features_rgb[y] = ctx.momentum * ctx.features_rgb[y] + (1. - ctx.momentum) * x
            ctx.features_rgb[y] /= ctx.features_rgb[y].norm()

        for x, y in zip(inputs_nir, targets):
            ctx.features_nir[y] = ctx.momentum * ctx.features_nir[y] + (1. - ctx.momentum) * x
            ctx.features_nir[y] /= ctx.features_nir[y].norm()

        return grad_inputs1, grad_inputs2, None, None, None, None

def cm(inputs_rgb, inputs_nir, indexes, features_rgb, features_nir, momentum):
    return CM.apply(inputs_rgb, inputs_nir, indexes, features_rgb, features_nir, torch.Tensor([momentum]).to(inputs_rgb.device))


class ClusterMemory_300(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory_300, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features_rgb', torch.zeros(num_samples, num_features))
        self.register_buffer('features_nir', torch.zeros(num_samples, num_features))


    def forward(self, inputs_rgb, inputs_nir, targets):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_nir = F.normalize(inputs_nir, dim=1).cuda()

        outputs_rgb, outputs_nir = cm(inputs_rgb, inputs_nir, targets, self.features_rgb, self.features_nir, self.momentum)
        outputs_rgb /= self.temp
        outputs_nir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets)
        loss_nir = F.cross_entropy(outputs_nir, targets)

        return loss_rgb, loss_nir



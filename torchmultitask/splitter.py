"""
by Guillaume Bellec in 2023
https://github.com/guillaumeBellec/multitask
"""

import torch
from torch import nn
import copy
import numpy as np
from typing import List, Tuple, Optional


def grad_projection(g_list, epsilon):

    # reshape on 2 dimension to treat batch differently
    shp = g_list[0].shape
    g_list = [g.reshape(shp[0], -1) for g in g_list]
    n_tasks = len(g_list)

    n = len(g_list)
    assert n > 1, "It does not make sense to use this for 1 task or less."

    # PC grad implementation:
    projected_g_list = []

    for i in range(n):
        g_i = g_list[i].clone()
        for j in torch.randperm(n):
            if i == j: continue
            g_j = g_list[j]
            g_i = truncated_proj_a_on_b(g_i, g_j, epsilon)

        projected_g_list += [g_i]

    projected_g_list = [g.reshape(shp) for g in projected_g_list]
    return projected_g_list


class NormalizedMultiTaskSplitterFunction(torch.autograd.Function):
    """
    Auto grad function which enables to override a custom backward pass.
    Warning: this is not slowing down a bit the training. Maybe a C++ custom implementation would be more efficient?
    """

    @staticmethod
    def forward(ctx, x,
                num_copies : int,
                m, t,
                beta, epsilon,
                scale,
                dummy_normalizer, projection_variant):
        d = x.device
        assert num_copies == m.numel(), f"Error: expected {num_copies} tasks with momentum tensor has shape {m.numel()}"
        ctx.save_for_backward(m, t, beta, epsilon, scale, dummy_normalizer, projection_variant)
        return tuple([x for _ in range(num_copies)])

    @staticmethod
    def backward(ctx, *grads_list):
        m, t, beta, epsilon, scale, dummy_normalizer, projection_variant = ctx.saved_tensors
        n_tasks = m.numel()

        if projection_variant == 3:
            grads_list = grad_projection(grads_list, epsilon)

        grads = torch.stack(grads_list, dim=-1)  # stack on the last dimension
        g_shp = grads.shape
        batch_size = g_shp[0]

        if dummy_normalizer or batch_size == 0:
            # just scale with the constant
            grads_normalized = grads * scale
        else:

            n_dims = len(grads.shape) - 1  # exclude the
            assert n_dims > 0, f"got {len(grads_list)} or shape {[g.shape for g in grads_list]}"
            grad_squared = grads.square().sum(tuple(range(1,n_dims)))
            grad_squared = grad_squared.mean(0) # mean over batch after all other dims

            # assuming normalization component-wise after batch dimension
            assert grad_squared.shape == m.shape, f"got gradient norm: {grad_squared.shape} but momentum is: {m.shape}"

            task_mask : torch.Tensor = grad_squared != 0 # exact zero only is null gradient. Do not update momentum.
            grad_squared = torch.where(task_mask, grad_squared, m) # do not update momentum if it's masked

            # inplace update
            m *= beta
            m += (1-beta) * grad_squared

            # count of number of non-zero gradient in updates (not incremented if grad is null)
            t += torch.where(task_mask, torch.ones_like(t), torch.zeros_like(t))

            # de-bias the momentum
            t_non_zero_mask = t > 0
            t_ = torch.where(t_non_zero_mask, t, torch.ones_like(t)) #
            v = m / (1 - torch.pow(beta, t_)) # avoid division by zero
            denom = torch.sqrt(v).clamp(min=epsilon)
            grads_normalized = grads / denom * scale * t_non_zero_mask.float() # set to strict zero if t=0 (never larger than epsilon)

        g_list = [grads_normalized[... ,i] for i in range(n_tasks)]

        if projection_variant == 1:
            g_sum = sum(grad_projection(g_list, epsilon))
        elif projection_variant == 2:
            # WARNING: This variant in not recommended.
            # This projection variant comes from Sener et al. 2018
            # https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf
            # But somehow my implementation here performs extremely bad in the 2 MNIST task
            # if you find out why let me know: guillaume.bellec@epfl.ch
            if len(g_list) != 2: raise NotImplementedError()
            g_sum = min_norm_proj(g_list[0], g_list[1], epsilon)
        else:
            g_sum = sum(g_list)

        return (g_sum,
                None, # num copies
                None, None, # m, t
                None, None, # beta epsilon
                None, # task weights
                None, None # boolean flags
        )


class NormalizedMultiTaskSplitter(nn.Module):

    def __init__(self,
                 loss_weights_dict,
                 projection_variant=0,
                 dummy_normalizer=False,
                 beta=0.999,  # correspond to a ~10_000 iteration memory
                 epsilon=1e-12,
                 dtype=torch.float32,
                 ):
        """
        This class enables to train multitask model easily. See the GitHub page for more info:
        https://github.com/guillaumeBellec/multitask

        In short, it enables two Multi task tricks:
        - Automatic normalization of loss gradient to avoid fine-tuning the loss weights.
        - Projection à la PC grad to avoid cross-talk between the losses (I am not sure if this effect is very strong).

        Be careful, both normalization and projection assume dimension 0 is batch dimension.

        Recommended loss weights is 1 for each task this no-brainer solution avoids fine-tuning:
        all tasks have the same influence on the downstream weights.
        If you want to prioritize one loss, you could try loss weights between 0.1 or 10. But with the automated
        gradient scaling loss weights larger than this expected to be less useful than when adding gradient naively.

        Recommended projection variant are 0, 1 or 3.
        0 (default): means no gradient project, it's the fastest. The normalization is basically more useful.
        1: Projection in the spirit of PC Grad, may achieve a bit higher performance.
        3: Same as 1, but doing the projection before normalizing gradient. Maybe slightly better than 1 on MNIST not sure.

        Set dummy_normalizer to True if you want to use projections without normalization.
        The loss weights are still applied but without normalization of the gradient.
        This is maybe useful if you know that the loss scales are well-balanced and you want to gain a few percents.

        :param loss_weights_obj: Re-weights the gradients after normalization (see https://arxiv.org/abs/2210.13438)
        :param projection_variant: Projection variant (0, 1, 2, or 3)
        :param dummy_normalizer: Skip the normalization with gradient norm momentum
        :param beta: similar to Adam, this is the momentum for the norm.
        :param epsilon: epsilon to avoid zero division in the normalization
        :param dtype: data type for the momentum tensors.
        """
        super().__init__()
        self.projection_variant = torch.tensor(projection_variant)
        self.dummy_normalizer = torch.tensor(dummy_normalizer)

        self.loss_weights_dict = loss_weights_dict
        #print("normalized task weights: ", self.get_normalized_weight_dict())

        num_tasks = len(loss_weights_dict.keys())
        self.epsilon = epsilon
        self.beta = beta
        self.register_buffer("_t", torch.zeros((num_tasks,), dtype=torch.int))
        self.register_buffer("_grad_var", torch.zeros((num_tasks,), dtype=dtype))

        self._function = NormalizedMultiTaskSplitterFunction()

    def update_weight(self, task_name, new_w):
        assert task_name in self.loss_weights_dict.keys()
        # update the obj of weight values
        self.loss_weights_dict[task_name] = new_w

    def get_normalized_weight_list(self):
        loss_weight_obj = self.loss_weights_dict
        w_list = list(loss_weight_obj.values())
        assert all([w >= 0 for w in w_list]), f"got weights {loss_weight_obj} but expecting non-negative loss weights"
        s = sum(w_list)
        if s == 0: return w_list
        return [w / s for w in w_list]

    def get_normalized_weight_dict(self):
        w_list = self.get_normalized_weight_list()
        return dict([(key, w_list[i]) for i,key in enumerate(self.loss_weights_dict.keys())])

    def forward(self, x):

        if not isinstance(x, torch.Tensor):
            raise NotImplementedError()

        loss_weights_tensor = torch.tensor(self.get_normalized_weight_list(), dtype=x.dtype, device=x.device)
        beta = torch.tensor(self.beta, dtype=x.dtype, device=x.device)
        epsilon = torch.tensor(self.epsilon, dtype=x.dtype, device=x.device)
        num_copies = len(loss_weights_tensor)

        x_tuple = self._function.apply(
            x,
            num_copies,
            self._grad_var.data, self._t.data,
            beta, epsilon,
            loss_weights_tensor,
            self.dummy_normalizer, self.projection_variant)

        return dict([(k, x_tuple[i]) for i,k in enumerate(self.loss_weights_dict.keys())])

    def _detached_weighted_loss_dict(self, loss_dict):
        # weighting the losses here have almost no effect with this multi-task torchmultitask
        # the point is to have an interface more similar to Defossez's interface,

        assert isinstance(self.loss_weights_dict, dict)
        assert loss_dict.keys() == self.loss_weights_dict.keys()
        normalized_weight_dict = self.get_normalized_weight_dict()

        weighted_loss = {}
        for i, key in enumerate(normalized_weight_dict.keys()):
            m_i = self._grad_var[i]
            lbda_i = self.normalized_weight_dict[key]
            weighted_loss_i = loss_dict[key] * lbda_i / m_i
            weighted_loss[key] = weighted_loss_i.detach() + 0 * weighted_loss_i

        return weighted_loss

    def geometric_loss_coeff(self, loss_dict, key_j):
        # TODO: try reweight the inner gradient with something like the geometric loss coeff?
        detached_weighted_losses = self._detached_weighted_loss_dict(loss_dict)
        keys = self.get_normalized_weight_dict().keys()
        n = len(keys)

        prod = 1.0 / torch.pow(detached_weighted_losses[key_j], n-1)
        for key_i in keys:
                if key_i == key_j: continue
                prod *= detached_weighted_losses[key_i]
        prod = torch.pow(prod, 1/n)
        geom_loss_j = prod * loss_dict[key_j]
        return geom_loss_j


def normalize(b, epsilon=1e-16):
    u = b / torch.sqrt((b * b).sum(1, keepdim=True)).clip(min=epsilon)
    return u

def norm_squared(a):
    return torch.square(a).sum(1, keepdim=True)


def min_norm_proj(a,b, epsilon=1e-10):

    gamma = ((b-a) * b).sum(1, keepdim=True) / (b-a).square().sum(1,keepdim=True).clip(min=epsilon**2)
    gamma = gamma.clip(min=0,max=1)
    c = a * gamma + (1 - gamma) * b
    return c, gamma


def truncated_proj_a_on_b(a,b, epsilon=1e-16):
    u = normalize(b, epsilon)
    prob_a_on_b = - torch.relu(- (a * u).sum(1, keepdim=True)) * u
    #a_proj_on_b = -torch.relu(- (a * u).sum(1, keepdim=True)) * u
    return a - prob_a_on_b


def min_norm_proj(a,b, epsilon=1e-16):
    gamma = ((b - a) * b).sum(1, keepdim=True) / norm_squared(b-a).clip(min=epsilon)
    gamma = gamma.clip(min=0,max=1)
    c = a * gamma + (1 - gamma) * b
    return c
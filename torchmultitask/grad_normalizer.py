import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class GradientNormalizingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, m : Tensor, t : int, beta : float, epsilon : float, scale: float, dummy_normalizer : bool):
        d = x.device
        ctx.save_for_backward(m, t, torch.tensor(beta,device=d), torch.tensor(epsilon,device=d), torch.tensor(scale,device=d), torch.tensor(dummy_normalizer))
        return x

    @staticmethod
    def backward(ctx, grads):

        m, t, beta, epsilon, scale, dummy_normalizer = ctx.saved_tensors

        if dummy_normalizer:
            return scale * grads, None, None, None, None, None, None

        g_shp = grads.shape
        batch_size = g_shp[0]
        if batch_size == 0:
            return scale * grads, None, None, None, None, None, None

        if torch.all(grads == 0):
            return scale * grads, None, None, None, None, None, None

        if len(m.shape) == 1 and m.shape[0] == 1:
            # just normalize the gradient norm
            grad_squared = (grads * grads).sum()/batch_size
        else:
            grad_squared = (grads * grads)
            while len(grad_squared.shape) > len(m.shape)+1:
                grad_squared = grad_squared.sum(-1)

            grad_squared = grad_squared.mean(0) # mean over batch after all other dims
            # assuming normalization component-wise after batch dimension
            assert grad_squared.shape == m.shape, "got gradient with shape: {} variance accumulated: {} ".format(grad_squared.shape, m.shape)

        if grad_squared.sum() == 0.: # strict zero no update of the momentum
            return torch.zeros_like(grads), None, None, None, None, None, None

        m *= beta
        m += (1-beta) * grad_squared
        t += 1

        v = m / (1 - torch.pow(beta,t))
        denom = torch.sqrt(v)
        denom = denom.clamp(min=epsilon)
        g = grads / denom

        return g * scale, None, None, None, None, None, None


class GradientNormalizer(nn.Module):

    def __init__(self, normalizer_shape=[1], beta=0.999, epsilon=1e-16, dtype=torch.float32, scale=1., dummy_normalizer=False):
        super(GradientNormalizer,self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("_t", torch.zeros([1], dtype=torch.int))
        self.register_buffer("_grad_var", torch.zeros(normalizer_shape, dtype=dtype))
        self.function = GradientNormalizingFunction()
        self.dummy_normalizer = dummy_normalizer
        self.scale=scale

    def forward(self, x):
        x = self.function.apply(x, self._grad_var.data, self._t.data, self.beta, self.epsilon, self.scale, self.dummy_normalizer)
        return x


class NormalizedMultiTaskSplitterSlow(nn.Module):

    def __init__(self, loss_weight_dict, feature_shape=[1],
                 beta=0.999, epsilon=1e-8, dtype=torch.float32,
                 dummy_normalizer=False,
                 ):

        super(NormalizedMultiTaskSplitterSlow,self).__init__()

        make_normalizer = lambda scale : GradientNormalizer(feature_shape,
                                                            beta=beta,
                                                            epsilon=epsilon,
                                                            dtype=dtype,
                                                            scale=scale,
                                                            dummy_normalizer=dummy_normalizer)

        self.loss_weight_dict = loss_weight_dict
        loss_weights = self.get_normalized_weight_list(loss_weight_dict)
        self.normalizers = nn.ModuleList([make_normalizer(w) for w in loss_weights])
        self.num_copies = len(loss_weights)

    def update_weight(self, index, new_w):

        # update the obj of weight values
        self.loss_weight_dict[index] = new_w

        # refresh normalized list and update normalizer scale attribute
        loss_weights = self.get_normalized_weight_list(self.loss_weights_obj, self.num_copies)
        for w, normalizer in zip(loss_weights,self.normalizers):
            assert isinstance(normalizer, GradientNormalizer)
            normalizer.scale = w

    def get_normalized_weight_list(self, loss_weight_dict):
        w_list = list(loss_weight_dict.values())

        for w in w_list:
            assert w >= 0, f"Only admits positive weights and got {loss_weight_dict}"

        s = sum(w_list)
        if s == 0: return w_list
        return [w / s for w in w_list]

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise NotImplementedError()

        x_tuple = tuple([x for _ in range(self.num_copies)])
        x_list = [normalizer(x) for x, normalizer in zip(x_tuple, self.normalizers)]

        pair_list = []
        for k,key in enumerate(self.loss_weight_dict.keys()):
            pair_list += [(key, x_list[k])]
        return dict(pair_list)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.x_target1 = torch.randn([2])
            self.x_target2 = torch.randn([2])
            loss_weights = {"task_1": 1, "task_2":1}
            self.splitter = NormalizedMultiTaskSplitterSlow(loss_weights, dummy_normalizer=False)

        def forward(self, x):
            x_dict = self.splitter(x)

            loss1 = torch.square(x_dict["task_1"] - self.x_target1).mean()
            loss2 = torch.square(x_dict["task_2"] - self.x_target2).mean() * 1000 # fake unbalanced loss

            return loss1 + loss2

    x_list = []
    x = nn.Parameter(torch.randn([2]))
    lr = 3e-3
    model = TestModel()

    for t in range(1000):
        loss = model(x)
        g, = torch.autograd.grad(loss, [x])

        x_list += [x.detach().cpu().numpy().copy()]
        x.data = x - lr * g

    x_list = np.array(x_list)
    plt.scatter(x_list[:,0], x_list[:, 1], c=np.linspace(0, 1, x_list.shape[0]), marker=".")
    plt.scatter(model.x_target1[0], model.x_target1[1], c="green")
    plt.scatter(model.x_target2[0], model.x_target2[1], c="red")
    plt.show()
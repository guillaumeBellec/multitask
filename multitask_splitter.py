import torch
from torch import jit
from torch import nn
import numpy as np
from torch import Tensor


class MultiTaskSplitterFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, num_copies : int, symmetric : bool):

        ctx.save_for_backward(torch.tensor(symmetric))
        output_list = [z for _ in range(num_copies)]
        return tuple(output_list)

    @staticmethod
    def backward(ctx, *g_list):

        use_symmetric = ctx.saved_tensors
        shp = g_list[0].shape
        g_list = [g.reshape(g.shape[0], -1) for g in g_list]

        n = len(g_list)

        # Orthogonalization of the gradients
        projected_g_list = []
        for i in range(n):
            g = g_list[i]
            for j in range(n if use_symmetric else i):
                g = truncated_proj_a_on_b(g, g_list[j])
            projected_g_list += [g]

        g = sum(projected_g_list)

        g = g.reshape(shp)
        return g, None, None, None


class MultiTaskSplitter(nn.Module):

    def __init__(self, num_copies, symmetric=True, random_order=True):
        super(MultiTaskSplitter, self).__init__()

        self.num_copies = num_copies
        self.symmetric = symmetric
        self.random_order = random_order if not symmetric else False
        self.function = MultiTaskSplitterFunction()

    def forward(self, x, perm=None):
        """
        If x is a Tensor it returns a tuple of copies. In the backward pass, the different gradients are
        (normalized if do_normalized=True and) projected on each other to avoid contradictory dimensions.

        If a list, a dict or a tuple is given, it returns tuple of lists, tuples of dicts.

        :param x: Tensor of list/dict/tuple of tensors
        :param num_copies: number of copies per tuple
        :param perm: order to specify loss priority
        :return:
        """
        num_copies = self.num_copies

        if perm is None:
            if self.random_order:
                perm = torch.randperm(num_copies)
            else:
                perm = torch.arange(num_copies)

        # When we can this for a list, it returns a tuple of list
        if isinstance(x, dict):
            res_as_list = self.forward(list(x.values()), perm)
            res_as_dicts = [{} for _ in range(num_copies)]
            for j in range(num_copies):
                for k, key in enumerate(x.keys()):
                    res_as_dicts[j][key] = res_as_list[j][k]

            return tuple(res_as_dicts)

        if isinstance(x, list):
            full_results = [[] for _ in range(num_copies)]
            for x_i in x:
                res = self.forward(x_i, perm)
                for j in range(num_copies):
                    full_results[j] += [res[j]]

            return tuple(full_results)

        if isinstance(x, tuple):
            return tuple([tuple(y) for y in self.forward(list(x), perm)])

        if x is None:
            return tuple([None for _ in range(num_copies)])

        assert(isinstance(x, torch.Tensor)), "got unexpected type: {}".format(type(x))

        x_copies = self.function.apply(x, num_copies, self.symmetric)
        x_copies = tuple([x_copies[i] for i in perm])
        return x_copies


class GradientNormalizingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, m : Tensor, t : int, beta : float, epsilon : float):

        ctx.save_for_backward(m, t, torch.tensor(beta), torch.tensor(epsilon))
        return x

    @staticmethod
    def backward(ctx, grads):

        m, t, beta, epsilon = ctx.saved_tensors

        grad_squared = (grads * grads).mean(0) # always, mean over batch dimension

        if len(m.shape) == 1 and m.shape[0] == 1:
            # just normalize the gradient norm
            grad_squared = grad_squared.sum()
        else:
            while len(grad_squared.shape) > len(m.shape):
                grad_squared = grad_squared.mean(0)
            # assuming normalization component-wise after batch dimension
            assert grad_squared.shape == m.shape, "got gradient with shape: {} variance accumulated: {} ".format(grad_squared.shape, m.shape)

        if t[0] == 0:
            m.copy_(grad_squared)
        else:
            m *= beta
            m += (1-beta) * grad_squared

        t += 1

        v = m / (1 - torch.pow(beta,t))
        g = grads / torch.sqrt(v).clip(min=epsilon)
        return g, None, None, None, None


class GradientNormalizer(nn.Module):

    def __init__(self, feature_shape=[1], beta=0.999, epsilon=1e-16, dtype=torch.float32):
        super(GradientNormalizer,self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("_t", torch.zeros([1], dtype=torch.int))
        self.register_buffer("_grad_var", torch.zeros(feature_shape, dtype=dtype))
        self.function = GradientNormalizingFunction()

    def forward(self, x):
        x = self.function.apply(x, self._grad_var.data, self._t.data, self.beta, self.epsilon)
        return x


class NormalizedMultiTaskSplitter(nn.Module):

    def __init__(self, num_copies, feature_shape=[1], symmetric=True, random_order=True, beta=0.999, epsilon=1e-16, dtype=torch.float32):
        super(NormalizedMultiTaskSplitter,self).__init__()
        self.num_copies = num_copies
        self.splitter = MultiTaskSplitter(num_copies=num_copies, random_order=random_order, symmetric=symmetric)
        make_normalizer = lambda : GradientNormalizer(feature_shape, beta, epsilon, dtype)
        self.normalizers = nn.ModuleList([make_normalizer() for _ in range(num_copies)])
        #self.out_normalizer = GradientNormalizer(feature_shape, beta, epsilon, dtype) if feature_shape != [1] else nn.Identity()

    def forward(self, x):
        #x = self.out_normalizer(x)
        x_tuple = self.splitter.forward(x)
        x_list = [normalizer(x) for x, normalizer in zip(x_tuple, self.normalizers)]
        return tuple(x_list)


def forward_backward_split(z_forward, z_backward):
    return (z_forward - z_backward).detach() + z_backward


def truncated_proj_a_on_b(a,b, epsilon=1e-16):
    u = b / torch.sqrt((b * b).sum(1, keepdim=True)).clip(min=epsilon)
    a_proj_on_b = -torch.relu(- (a * u).sum(1, keepdim=True)) * u
    return a - a_proj_on_b


def norm_squared(a):
    return torch.square(a).sum(1, keepdim=True)


def min_norm_proj(a,b, epsilon=1e-16):

    gamma = ((b - a) * b).sum(1, keepdim=True) / norm_squared(b-a).clip(min=epsilon)
    gamma = gamma.clip(min=0,max=1)
    c = a * gamma + (1 - gamma) * b
    return c


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.x_target1 = torch.randn([2])
            self.x_target2 = torch.randn([2])
            self.splitter = NormalizedMultiTaskSplitter(2, [2])
            #self.splitter = MultiTaskSplitter(2,0)

        def forward(self, x):
            xs = self.splitter(x)
            loss1 = torch.square(xs[0] - self.x_target1).mean()
            loss2 = torch.square(xs[1] - self.x_target2).mean() * 1000

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


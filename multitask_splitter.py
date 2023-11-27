import torch
from torch import jit
from torch import nn
import numpy as np
from torch import Tensor
from utils.flatten_and_restore import flatten_structure, restore_structure


class MultiTaskSplitterFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, num_copies : int, symmetric : bool, random_order : bool):

        ctx.save_for_backward(torch.tensor(symmetric), torch.tensor(random_order))
        output_list = [z for _ in range(num_copies)]
        return tuple(output_list)

    @staticmethod
    def backward(ctx, *g_list):

        use_symmetric, random_order = ctx.saved_tensors
        shp = g_list[0].shape
        g_list = [g.reshape(g.shape[0], -1) for g in g_list]

        n = len(g_list)
        if use_symmetric and random_order:
            raise NotImplementedError()

        # Orthogonalization of the gradients
        projected_g_list = []
        if random_order:
            g_list = [g_list[i] for i in torch.randperm(n)]

        for i in range(n):
            g_i = g_list[i]
            for j in range(n if use_symmetric else i):
                # make sure that g always has zero scalar product with g_list[j]
                g_i = truncated_proj_a_on_b(g_i, g_list[j])

            projected_g_list += [g_i]

        g = sum(projected_g_list)

        g = g.reshape(shp)
        return g, None, None, None


class MultiTaskSplitter(nn.Module):

    def __init__(self, num_copies, symmetric=False, random_order=True):
        super(MultiTaskSplitter, self).__init__()

        self.num_copies = num_copies
        self.symmetric = symmetric
        self.random_order = random_order
        if symmetric and random_order:
            raise NotImplementedError()
        self.function = MultiTaskSplitterFunction()

    def forward(self, x):
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

        if x is None:
            return tuple([None for _ in range(num_copies)])

        assert(isinstance(x, torch.Tensor)), "got unexpected type: {}".format(type(x))

        x_copies = self.function.apply(x, num_copies, self.symmetric, self.random_order)

        return x_copies


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


class NormalizedMultiTaskSplitter(nn.Module):

    def __init__(self, loss_weights_obj=None, num_copies=None, feature_shape=[1],
                 symmetric=False, random_order=True, skip_projections=True, dummy_normalizer=False,
                 beta=0.9999, epsilon=1e-8, dtype=torch.float32,
                 ):

        super(NormalizedMultiTaskSplitter,self).__init__()
        self.skip_projections = skip_projections

        make_normalizer = lambda scale : GradientNormalizer(feature_shape,
                                                            beta=beta,
                                                            epsilon=epsilon,
                                                            dtype=dtype,
                                                            scale=scale,
                                                            dummy_normalizer=dummy_normalizer)

        self.loss_weights_obj = loss_weights_obj
        loss_weights = self.get_normalized_weight_list(loss_weights_obj, num_copies)
        self.normalizers = nn.ModuleList([make_normalizer(w) for w in loss_weights])
        self.num_copies = num_copies if num_copies is not None else len(loss_weights)

        if not random_order and not symmetric:
            assert loss_weights[0] >= loss_weights[1], f"Expecting loss weight in descending order when the tasks have an order of priority: but we have {loss_weights} (loss_weights_obj={loss_weights_obj})"

        if not skip_projections:
            self.splitter = MultiTaskSplitter(num_copies=self.num_copies, random_order=random_order, symmetric=symmetric)

    def update_weight(self, index, new_w):

        # update the obj of weight values
        self.loss_weights_obj[index] = new_w

        # refresh normalized list and update normalizer scale attribute
        loss_weights = self.get_normalized_weight_list(self.loss_weights_obj, self.num_copies)
        for w, normalizer in zip(loss_weights,self.normalizers):
            assert isinstance(normalizer, GradientNormalizer)
            normalizer.scale = w

    def get_normalized_weight_list(self, loss_weight_obj, num_copies):
        if loss_weight_obj is None: loss_weight_obj = [1.] * num_copies
        w_list = flatten_structure(loss_weight_obj)

        for w in w_list:
            assert w >= 0, f"Only admits positive weights and got {loss_weight_obj}"

        s = sum(w_list)
        if s == 0: return w_list
        return [w / s for w in w_list]

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise NotImplementedError()

        if self.skip_projections:
            x_tuple = tuple([x for _ in range(self.num_copies)])
        else:
            x_tuple = self.splitter.forward(x)
        x_list = [normalizer(x) for x, normalizer in zip(x_tuple, self.normalizers)]
        if self.loss_weights_obj is None:
            return tuple(x_list)
        else:
            return restore_structure(self.loss_weights_obj, x_list)

    def weighted_loss(self, loss_dict):
        if isinstance(loss_dict, dict):
            assert isinstance(self.loss_weights_obj, dict)
            assert loss_dict.keys() == self.loss_weights_obj.keys()

            return sum([self.loss_weights_obj[k] * loss_dict[k] for k in loss_dict.keys()])

    def backward(self, obj, *args):
        s = self.weighted_loss(obj)
        s.backward()


def forward_backward_split(z_forward, z_backward):
    return (z_forward - z_backward).detach() + z_backward


def truncated_proj_a_on_b(a,b, epsilon=1e-16):
    if torch.all(b == 0.): return a
    if torch.all(a == 0.): return torch.zeros_like(a)

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
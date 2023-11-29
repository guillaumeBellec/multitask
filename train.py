import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from utils.multi_mnist_dataset import MultiMNIST
from net.lenet import LeNet5Encoder, MLP
from torchmultitask.pcgrad import PCGrad
from utils.logging import create_logger
from torchmultitask.splitter import NormalizedMultiTaskSplitter
import argparse

# ------------------ CHANGE THE CONFIGURATION -------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='./dataset')
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.1)
parser.add_argument("--lr_phase_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_fc", type=int, default=128)
parser.add_argument("--balanced", type=int, default=0)
parser.add_argument("--simulation_name", type=str, default="normalized-splitter")
args = parser.parse_args()

TASKS = ['R', 'L']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
task_weight_dict = {'R': 1., 'L': 1.}

splitter_obj_dict = {
    "single-task": None,
    "pcgrad": None,
    "summed-loss": None,

    "normalized-splitter": NormalizedMultiTaskSplitter(task_weight_dict),
    "normalized-project1-splitter": NormalizedMultiTaskSplitter(task_weight_dict, projection_variant=1),
    "normalized-project2-splitter": NormalizedMultiTaskSplitter(task_weight_dict, projection_variant=2),
    "normalized-project3-splitter": NormalizedMultiTaskSplitter(task_weight_dict, projection_variant=3),
    "project1-splitter": NormalizedMultiTaskSplitter(task_weight_dict, dummy_normalizer=True, projection_variant=1),
    "project2-splitter": NormalizedMultiTaskSplitter(task_weight_dict, dummy_normalizer=True, projection_variant=2),
    "project3-splitter": NormalizedMultiTaskSplitter(task_weight_dict, dummy_normalizer=True, projection_variant=3),
}
# ---------------------------------------------------------


accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]
logger = create_logger('Main')

global_transformer = transforms.Compose(
        [transforms.Normalize((0.1307, ), (0.3081, ))])

CE = nn.CrossEntropyLoss(label_smoothing=0.1)

def stop_grad(x):
    return x.detach() + x * 0

def change_learning_rate(optim, new_lr, verbose=True):
    old_lr = optim.param_groups[0]['lr']
    if old_lr != new_lr:
        if verbose: print(f"set learning rate: {old_lr:0.5f} -> {new_lr:0.5f}")
        optim.param_groups[0]['lr'] = new_lr


def run(simulation_name):
    splitter = splitter_obj_dict[simulation_name]
    if splitter is not None: splitter = splitter.to(DEVICE)

    logger.info(f"--> Starting training for: {simulation_name}")
    if not args.balanced:
        simulation_name += "-imbalanced"

    results = {
               'train_loss_R': [],
               'train_loss_L': [],

               'test_acc_R': [],
               'test_acc_L': [],

               'test_loss_R': [],
               'test_loss_L': [],
               }

    train_dst = MultiMNIST(args.path,
                           train=True,
                           download=True,
                           transform=global_transformer,
                           multi=True)

    train_loader = torch.utils.data.DataLoader(train_dst,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    val_dst = MultiMNIST(args.path,
                         train=False,
                         download=True,
                         transform=global_transformer,
                         multi=True)

    val_loader = torch.utils.data.DataLoader(val_dst,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1)
    nets = {
        'rep': LeNet5Encoder(args.n_fc).to(DEVICE),
        'L': MLP(args.n_fc).to(DEVICE),
        'R': MLP(args.n_fc).to(DEVICE)
    }

    param = [p for v in nets.values() for p in list(v.parameters())]
    adam = torch.optim.Adam(param, lr=args.lr)
    optimizer = PCGrad(adam) if simulation_name.startswith("pcgrad") else adam

    mom = 0.99
    train_loss_L = None
    train_loss_R = None

    for ep in range(args.num_epochs):

        if ep % args.lr_phase_length == 0 and ep > 0:
            factor = args.lr_decay ** (ep // args.lr_phase_length)
            change_learning_rate(adam, args.lr * factor)

        for net in nets.values():
            net.train()
        epoch_len = len(train_loader)
        for k_step,batch in enumerate(train_loader):

            if ep == 0:
                change_learning_rate(adam, args.lr * k_step / epoch_len, verbose=k_step % 200 == 1)

            optimizer.zero_grad()
            img, label_l, label_r = to_dev(batch, DEVICE)
            rep = nets['rep'](img)

            if "splitter" in simulation_name:
                rep_dict = splitter.forward(rep)
                out_l = nets['L'](rep_dict["L"])
                out_r = nets['R'](rep_dict["R"])
            else:
                out_l = nets['L'](rep)
                out_r = nets['R'](rep)

            loss_r = CE(out_r, label_r)
            loss_l = CE(out_l, label_l)

            if not args.balanced:
                loss_r *= 1000. # create imbalance artificially

            losses = [loss_l, loss_r]

            train_loss_R = loss_r.mean().item() if train_loss_R is None else mom * train_loss_R + (1-mom) * loss_r.mean().item()
            train_loss_L = loss_l.mean().item() if train_loss_L is None else mom * train_loss_L + (1-mom) * loss_l.mean().item()

            if "pcgrad" in simulation_name: optimizer.pc_backward(losses)
            elif "single-task" in simulation_name: losses[0].backward()
            else: sum(losses).backward()

            optimizer.step()

        # TESTING
        losses, acc = [], []
        for net in nets.values():
            net.eval()
        with torch.no_grad():
            for batch in val_loader:
                img, label_l, label_r = to_dev(batch, DEVICE)
                rep = nets['rep'](img)
                out_l = nets['L'](rep)
                out_r = nets['R'](rep)

                losses.append([
                    F.nll_loss(out_l, label_l).item(),
                    F.nll_loss(out_r, label_r).item()
                ])
                acc.append(
                    [accuracy(out_l, label_l).item(),
                     accuracy(out_r, label_r).item()])
        losses, acc = np.array(losses), np.array(acc)
        logger.info('{}: epoches {}/{}: loss test (left, right) = {:5.4f}, {:5.4f} \t train {:5.4f}, {:5.4f}'.format(
            simulation_name, ep, args.num_epochs, losses[:,0].mean(), losses[:1].mean(), train_loss_L, train_loss_R))
        logger.info('{}: epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
            simulation_name, ep, args.num_epochs, acc[:,0].mean(), acc[:,1].mean()))

        results['train_loss_L'] += [float(train_loss_L)]
        results['train_loss_R'] += [float(train_loss_R)]

        results['test_acc_L'] += [float(acc[:,0].mean())]
        results['test_acc_R'] += [float(acc[:,1].mean())]

        results['test_loss_L'] += [float(losses[:,0].mean())]
        results['test_loss_R'] += [float(losses[:,1].mean())]

    date = datetime.now()
    file_name = f"{date.year}_{date.month:02d}_{date.day:02d}_{date.hour:02d}_{date.minute:02d}_{date.second:02d}_{date.microsecond}_{simulation_name}.json"
    with open("results/" + file_name, "w") as f:
        json.dump(results,f, indent=4)

run(args.simulation_name)
import json
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
from data.multi_mnist import MultiMNIST
from net.lenet import LeNet5Encoder, MLP
from pcgrad import PCGrad
from utils import create_logger
from multitask_splitter import MultiTaskSplitter, NormalizedMultiTaskSplitter

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset'
NUM_EPOCHS = 80
LR = 0.001
LR_decay = 0.1
LR_phase_length = 20
BATCH_SIZE = 32
n_fc = 128
balanced_loss = True
TASKS = ['R', 'L']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

configs = {
    #"single-task": None,
    "pcgrad": None,
    "summed-loss": None,

    "normd-global-splitter": NormalizedMultiTaskSplitter(2,[1], symmetric=False),
    "splitter-truncated": MultiTaskSplitter(num_copies=2, symmetric=False),
    "normd-splitter": NormalizedMultiTaskSplitter(2, [n_fc], symmetric=False),

    "normd-global-splitter-sym": NormalizedMultiTaskSplitter(2, [1], symmetric=True),
    "splitter-truncated-sym": MultiTaskSplitter(num_copies=2, symmetric=True),
    "normd-splitter-sym": NormalizedMultiTaskSplitter(2,[n_fc], symmetric=True),
}
# ---------------------------------------------------------


accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]
logger = create_logger('Main')

global_transformer = transforms.Compose(
        [transforms.Normalize((0.1307, ), (0.3081, ))])

CE = nn.CrossEntropyLoss(label_smoothing=0.1)

def change_learning_rate(optim, new_lr):
    old_lr = optim.param_groups[0]['lr']
    if old_lr != new_lr:
        print(f"set learning rate: {old_lr} -> {new_lr}")
        optim.param_groups[0]['lr'] = new_lr

for simulation_name, splitter in configs.items():

    logger.info(f"--> Starting training for: {simulation_name}")
    if not balanced_loss:
        simulation_name += "-imbalanced"

    results = {
               'train_loss_R': [],
               'train_loss_L': [],

               'test_acc_R': [],
               'test_acc_L': [],

               'test_loss_R': [],
               'test_loss_L': [],
               }

    train_dst = MultiMNIST(PATH,
                           train=True,
                           download=True,
                           transform=global_transformer,
                           multi=True)
    train_loader = torch.utils.data.DataLoader(train_dst,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=4)

    val_dst = MultiMNIST(PATH,
                         train=False,
                         download=True,
                         transform=global_transformer,
                         multi=True)
    val_loader = torch.utils.data.DataLoader(val_dst,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1)
    nets = {
        'rep': LeNet5Encoder(n_fc).to(DEVICE),
        'L': MLP(n_fc).to(DEVICE),
        'R': MLP(n_fc).to(DEVICE)
    }

    param = [p for v in nets.values() for p in list(v.parameters())]
    adam = torch.optim.Adam(param, lr=LR)
    optimizer = PCGrad(adam) if simulation_name == "pcgrad" else adam

    mom = 0.99
    train_loss_L = None
    train_loss_R = None

    for ep in range(NUM_EPOCHS):

        if ep % LR_phase_length == 0:
            factor = LR_decay ** (ep // LR_phase_length)
            change_learning_rate(adam, LR * factor)

        for net in nets.values():
            net.train()
        for batch in train_loader:
            optimizer.zero_grad()
            img, label_l, label_r = to_dev(batch, DEVICE)
            rep = nets['rep'](img)

            if "splitter" in simulation_name:
                rep_l, rep_r = splitter.forward(rep)
            else:
                rep_l, rep_r = rep, rep

            out_l = nets['L'](rep_l)
            out_r = nets['R'](rep_r)

            loss_r = CE(out_r, label_r)
            loss_l = CE(out_l, label_l)

            if not balanced_loss:
                loss_r *= 1000.

            losses = [loss_l, loss_r]

            train_loss_R = loss_r.mean().item() if train_loss_R is None else mom * train_loss_R + (1-mom) * loss_r.mean().item()
            train_loss_L = loss_l.mean().item() if train_loss_L is None else mom * train_loss_L + (1-mom) * loss_l.mean().item()

            if simulation_name == "pcgrad": optimizer.pc_backward(losses)
            elif simulation_name == "single-task": losses[0].backward()
            else: sum(losses).backward()

            optimizer.step()

        losses, acc = [], []
        for net in nets.values():
            net.eval()
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
        logger.info('epoches {}/{}: loss test (left, right) = {:5.4f}, {:5.4f} \t train {:5.4f}, {:5.4f}'.format(
            ep, NUM_EPOCHS, losses[:,0].mean(), losses[:1].mean(), train_loss_L, train_loss_R))
        logger.info('epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
                ep, NUM_EPOCHS, acc[:,0].mean(), acc[:,1].mean()))

        results['train_loss_L'] += [float(train_loss_L)]
        results['train_loss_R'] += [float(train_loss_R)]

        results['test_acc_L'] += [float(acc[:,0].mean())]
        results['test_acc_R'] += [float(acc[:,1].mean())]

        results['test_loss_L'] += [float(losses[:,0].mean())]
        results['test_loss_R'] += [float(losses[:,1].mean())]

    date = datetime.now()
    file_name = f"{date.year}_{date.month:02d}_{date.day:02d}_{date.hour:02d}_{date.minute:02d}_{date.second:02d}_{date.microsecond}_{simulation_name}.json"
    with open("results_with_schedule/" + file_name, "w") as f:
        json.dump(results,f, indent=4)